"""Why does the terrain curriculum stop rising? Measure the criterion itself.

`terrain_levels_vel` promotes an env when it walked more than 4 m from its spawn
(half an 8 m tile) and demotes it when it walked less than half of
|command| x episode_length. Under the flat tasks' bucket command distribution a
lot of episodes are standing / slow / pure-turn, where those distances are out
of reach whatever the terrain. This bench runs one episode per env on a pinned
(row, column), with the task's real command distribution, and reports the
distance walked against both thresholds, split by terrain column and by the
command bucket the env drew.

usage: python tools/rough_diag.py [--task T] [--ckpt PATH] [--level 0,1]
                                  [--e 24] [--reps 3] [--t 20]
"""

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Playground-Velocity-Rough-Dgv2-M22-Unitree-Go2W")
ap.add_argument(
  "--ckpt",
  default="logs/rsl_rl/playground_go2w_rough_dgv2_m22_velocity/"
  "2026-11-13_11-54-03_rough-v1-ws-a8ea1db",
)
ap.add_argument("--iter", default=None, help="checkpoint iteration; default = latest")
ap.add_argument("--level", default="0,1", help="comma-separated rows to test")
ap.add_argument("--e", type=int, default=24, help="envs per (row, column)")
ap.add_argument("--reps", type=int, default=3, help="episodes per env")
ap.add_argument("--t", type=float, default=20.0, help="episode length, s")
args = ap.parse_args()

if args.iter:
  ck = Path(args.ckpt) / f"model_{args.iter}.pt"
else:
  ck = sorted(
    Path(args.ckpt).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
  )[-1]
print(f"checkpoint {ck}")
policy = gb._load_policy(ck)

levels = [int(x) for x in args.level.split(",")]
E = args.e

cfg = load_env_cfg(args.task, play=True)
gen = cfg.scene.terrain.terrain_generator
cols = list(gen.sub_terrains)
cfg.scene.num_envs = len(cols) * len(levels) * E
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.observations["actor"].enable_corruption = False
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)

N = env.num_envs
terrain = env.scene.terrain
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
# env index -> (level, column)
lv = torch.repeat_interleave(torch.tensor(levels, device=D), E * len(cols))
col = torch.arange(len(cols), device=D).repeat(len(levels) * E)
idx = torch.arange(N, device=D)
row_of = lv
col_of = col

steps = int(args.t / env.step_dt)
rows_out = []  # (level, col, bucket, distance, |cmd|, died)


def place():
  env.reset()
  terrain.terrain_levels[:] = lv
  terrain.terrain_types[:] = col
  terrain.env_origins[:] = terrain.terrain_origins[lv, col]
  st = robot.data.default_root_state.clone()
  st[:, :3] = terrain.env_origins + torch.tensor([0.0, 0.0, 0.42], device=D)
  yaw = (torch.rand(N, device=D) * 2 - 1) * math.radians(5)
  st[:, 3:7] = torch.stack(
    [
      torch.cos(yaw / 2),
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      torch.sin(yaw / 2),
    ],
    1,
  )
  st[:, 7:] = 0.0
  robot.write_root_state_to_sim(st)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
  )
  term._resample_command(idx)
  obs, *_ = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=D))
  return obs


for _ in range(args.reps):
  obs = place()
  o = terrain.env_origins.clone()
  cmd = term.vel_command_b.clone()
  bucket = term.bucket.clone()
  fell = torch.zeros(N, dtype=torch.bool, device=D)
  with torch.no_grad():
    for _ in range(steps):
      obs, *_ = env.step(policy(obs["actor"]))
      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      fell |= tilt > math.radians(80)
  d = (robot.data.root_link_pos_w[:, :2] - o[:, :2]).norm(dim=1)
  rows_out.append(
    torch.stack(
      [
        row_of.float(),
        col_of.float(),
        bucket.float(),
        d,
        cmd[:, :2].norm(dim=1),
        fell.float(),
      ],
      dim=1,
    )
  )

R = torch.cat(rows_out).cpu()
lv_n, col_n, bk_n, dist, cmd_n, fell = (R[:, i] for i in range(6))
names = cols
BUCKETS = ("standing", "vx", "vy", "wz", "slow", "combo")
T = args.t

print(
  f"\nepisode {T:.0f}s, {args.reps} reps/env, mean distance walked (m). "
  "up: > 4.0 m, down: < |cmd| * T * 0.5"
)
print(
  f"\n{'column':22s} {'lvl':>3s} {'n':>5s} {'dist':>6s} {'up%':>6s} {'down%':>6s} {'fell%':>6s}"
)
for name in names:
  for L in levels:
    s = (lv_n == L) & (torch.tensor([names.index(name)]) == col_n)
    if not s.any():
      continue
    up = (dist[s] > 4.0).float().mean().item() * 100
    down = (dist[s] < cmd_n[s] * T * 0.5).float().mean().item() * 100
    print(
      f"{name:22s} {L:3d} {int(s.sum()):5d} {dist[s].mean():6.2f} "
      f"{up:6.0f} {down:6.0f} {fell[s].float().mean().item() * 100:6.0f}"
    )

print(
  f"\n{'bucket':22s} {'n':>5s} {'|cmd|':>6s} {'reach':>7s} {'dist':>6s} {'up%':>6s} {'down%':>6s}"
)
for i, b in enumerate(BUCKETS):
  s = bk_n == i
  if not s.any():
    continue
  print(
    f"{b:22s} {int(s.sum()):5d} {cmd_n[s].mean():6.2f} "
    f"{(cmd_n[s] * T).mean():7.2f} {dist[s].mean():6.2f} "
    f"{(dist[s] > 4.0).float().mean().item() * 100:6.0f} "
    f"{(dist[s] < cmd_n[s] * T * 0.5).float().mean().item() * 100:6.0f}"
  )

# Up/down outcome mix: what the curriculum would actually do this episode.
up = dist > 4.0
down = dist < cmd_n * T * 0.5
print(
  f"\nall envs: up {100 * up.float().mean():.0f}%  down {100 * down.float().mean():.0f}%  "
  f"stay {100 * (~up & ~down).float().mean():.0f}%"
)
for name in names:
  s = torch.tensor([names.index(name)]) == col_n
  print(
    f"  {name:22s} up {100 * up[s].float().mean():5.0f}%  "
    f"down {100 * down[s].float().mean():5.0f}%  "
    f"stay {100 * (~up[s] & ~down[s]).float().mean():5.0f}%"
  )

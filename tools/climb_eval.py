"""Ledge-climb benchmark for the Go2W Climb tasks.

Every env goes onto the climbing column, E envs per row, at the pit centre facing
+x (yaw jitter +-YAW_DEG), with a fixed command: vx = VX, heading 0. The step
height of each tile is read back from its origin (inverted pyramid with one
inner step: origin z = -2h). Per row it reports:

  tread%  got onto the first tread (Chebyshev distance > 2.15 m from centre)
  top%    got onto the top (distance > 3.15 m, i.e. both rises)
  fall%   tilted past 80 deg before reaching the top
  t_top   median time to the top (s), over the envs that made it

Policy noise is off (mean action), obs noise off unless NOISE=1, no pushes;
wheel friction is the env's startup draw (0.5-1.0).

usage: python tools/climb_eval.py <task> <run_glob>@<iter> [...]
env:   VX (0.5)  T (15 s)  E (16)  YAW_DEG (5)  NOISE (0: actor obs noise off, 1: on)
       HRANGE ("lo,hi": override the climbing step range, e.g. 0.45,0.60)
       CALF (calf effort limit override, N m)
"""

import contextlib
import io
import math
import os
import sys
import types
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
VX = float(os.environ.get("VX", "0.5"))
T = float(os.environ.get("T", "15"))
E = int(os.environ.get("E", "16"))
YAW = math.radians(float(os.environ.get("YAW_DEG", "5")))
NOISE = os.environ.get("NOISE", "0") == "1"
CLIMB = "high_platform"
LOGS = {
  "Playground-Velocity-Climb-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-Blind-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-H65-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-Blind-H65-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Stop-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-DC-H65-FlatPose-Bar-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-DC-H65-FlatPose-Bar2-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-TrunkTerm-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
  "Playground-Velocity-Climb-BlindNoBit-DC-H65-Unitree-Go2W": "playground_go2w_climb_blind_nobit_velocity",
}

task = sys.argv[1]
cfg = load_env_cfg(task, play=True)
rows = cfg.scene.terrain.terrain_generator.num_rows
cfg.scene.num_envs = rows * E
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.observations["actor"].enable_corruption = NOISE
if os.environ.get("HRANGE"):
  lo, hi = (float(x) for x in os.environ["HRANGE"].split(","))
  cfg.scene.terrain.terrain_generator.sub_terrains[CLIMB].step_height_range = (lo, hi)
if os.environ.get("CALF"):
  acts = cfg.scene.entities["robot"].articulation.actuators
  for a in acts:
    if a.target_names_expr == (".*_calf_joint",):
      a.effort_limit = float(os.environ["CALF"])
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
terrain = env.scene.terrain
robot = env.scene["robot"]
col = list(cfg.scene.terrain.terrain_generator.sub_terrains).index(CLIMB)
row_of = torch.arange(N, device=D) // E

term = env.command_manager.get_term("twist")


def _fixed(self, ids):
  self.vel_command_b[ids, 0] = VX
  self.vel_command_b[ids, 1:] = 0.0
  self.heading_target[ids] = 0.0
  self.is_heading_env[ids] = True
  self.is_standing_env[ids] = False
  self.is_world_env[ids] = False
  self.is_forward_env[ids] = False
  if hasattr(self, "_climb"):  # ClimbAwareBucketVelocityCommand: heading control on
    self._climb[ids] = True
    self._axis_pending[ids] = False


term._resample_command = types.MethodType(_fixed, term)


def place():
  env.reset()
  terrain.terrain_types[:] = col
  terrain.terrain_levels[:] = row_of
  terrain.env_origins[:] = terrain.terrain_origins[row_of, col]
  st = robot.data.default_root_state.clone()
  st[:, :3] = terrain.env_origins + torch.tensor([0.0, 0.0, 0.42], device=D)
  yaw = (torch.rand(N, device=D) * 2 - 1) * YAW
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
  term._resample_command(torch.arange(N, device=D))
  obs, *_ = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=D))
  return obs


def run(policy):
  obs = place()
  o = terrain.env_origins.clone()
  h = -o[:, 2] / 2
  steps = int(T / env.step_dt)
  tread = torch.zeros(N, dtype=torch.bool, device=D)
  top = torch.zeros_like(tread)
  fell = torch.zeros_like(tread)
  t_top = torch.full((N,), float("nan"), device=D)
  with torch.no_grad():
    for k in range(steps):
      obs, *_ = env.step(policy(obs["actor"]))
      p = robot.data.root_link_pos_w
      d = torch.maximum((p[:, 0] - o[:, 0]).abs(), (p[:, 1] - o[:, 1]).abs())
      z = p[:, 2] - o[:, 2]
      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      fell |= (tilt > math.radians(80)) & ~top
      tread |= (d > 2.15) & (z > h + 0.15) & ~fell
      new_top = (d > 3.15) & (z > 2 * h + 0.15) & ~fell & ~top
      t_top[new_top] = (k + 1) * env.step_dt
      top |= new_top
  return h, tread, top, fell, t_top


base = Path("logs/rsl_rl") / LOGS[task]
for spec in sys.argv[2:]:
  g, it = spec.split("@")
  ck = sorted(base.glob(g))[-1] / f"model_{it}.pt"
  h, tread, top, fell, t_top = run(gb._load_policy(ck))
  print(
    f"\n{ck.parent.name} @ {it}   VX {VX}  T {T}s  E {E}/row  yaw +-{math.degrees(YAW):.0f} deg  obs noise {NOISE}  HRANGE {os.environ.get('HRANGE', '-')}  CALF {os.environ.get('CALF', '-')}"
  )
  print("  row   h (m)        tread%  top%  fall%  t_top(s)")
  best = 0.0
  for r in range(rows):
    s = row_of == r
    tt = t_top[s & top]
    med = f"{tt.median().item():5.1f}" if len(tt) else "    -"
    rate = top[s].float().mean().item()
    if rate >= 0.8:
      best = max(best, h[s].max().item())
    print(
      f"  {r:3d}  {h[s].min().item():.3f}-{h[s].max().item():.3f}"
      f"  {100 * tread[s].float().mean():6.0f} {100 * rate:5.0f} {100 * fell[s].float().mean():6.0f}   {med}"
    )
  print(
    f"  overall top% {100 * top.float().mean():.0f}  fall% {100 * fell.float().mean():.0f}"
    f"  | highest row with top >= 80%: h <= {best:.3f} m"
  )

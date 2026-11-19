"""Monte-Carlo test of the key obstacle: a long bar 40 cm high, 40 cm wide.

N envs, each with its own random draw of the conditions an operator would
meet, all at once:
  speed      vx ~ U(0.3, 0.8) m/s
  approach   start yaw ~ U(-20, 20) deg (heading command 0, i.e. square to the bar)
  run-up     start ~ U(0.6, 2.0) m before the bar's near face
  wheels     friction ~ U(0.4, 1.0)
  sensing    actor observation noise on (training levels)
Climb bit on (the bar sits on the "high_platform" column, see bar_bench.py).
Pass = more than 0.5 m past the far face and back on the floor within T, never
tilted past 80 deg. Reports pass / fall / stuck with a 95% upper bound on the
fall rate (exact binomial; 0 falls in N gives 1 - 0.05^(1/N)).

usage: python tools/bar_mc.py <task> <run_glob>@<iter> [more specs ...]
env:   N (256)  H (0.40)  WIDTH (0.40)  T (16)
"""

import contextlib
import io
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402
from mjlab_playground.velocity.config.go2w_climb.climb_terrains import (  # noqa: E402
  BoxBarRingTerrainCfg,
)

D = "cuda:0"
gb.DEVICE = D
N = int(os.environ.get("N", "256"))
H = float(os.environ.get("H", "0.40"))
WIDTH = float(os.environ.get("WIDTH", "0.40"))
T = float(os.environ.get("T", "16"))
CLIMB = "high_platform"
RING = 2.0

task = sys.argv[1]
cfg = load_env_cfg(task, play=True)
gen = cfg.scene.terrain.terrain_generator
cfg.scene.terrain.terrain_generator = replace(
  gen,
  num_rows=1,
  num_cols=2,
  sub_terrains={
    "flat": replace(gen.sub_terrains["flat"], proportion=0.5),
    CLIMB: BoxBarRingTerrainCfg(
      proportion=0.5, heights=(H,), bar_width_range=(WIDTH, WIDTH), ring_distance=RING
    ),
  },
)
cfg.scene.num_envs = N
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.observations["actor"].enable_corruption = True
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
terrain = env.scene.terrain
robot = env.scene["robot"]
col = list(cfg.scene.terrain.terrain_generator.sub_terrains).index(CLIMB)
ids_all = torch.arange(N, device=D)

g0 = torch.Generator(device=D).manual_seed(1)
U = lambda lo, hi: lo + (hi - lo) * torch.rand(N, device=D, generator=g0)  # noqa: E731
vx = U(0.3, 0.8)
yaw = torch.deg2rad(U(-20.0, 20.0))
start = U(0.6, 2.0)
fric = U(0.4, 1.0)
wl = torch.tensor(robot.find_geoms(".*_wheel_collision")[0], device=D)
gid = robot.indexing.geom_ids.to(D)[wl].long()
env.sim.model.geom_friction[:, gid, 0] = fric[:, None]

term = env.command_manager.get_term("twist")


def _fixed(self, ids):
  self.vel_command_b[ids, 0] = vx[ids]
  self.vel_command_b[ids, 1:] = 0.0
  self.heading_target[ids] = 0.0
  self.is_heading_env[ids] = True
  self.is_standing_env[ids] = False
  self.is_world_env[ids] = False
  self.is_forward_env[ids] = False
  if hasattr(self, "_climb"):
    self._climb[ids] = True
    self._axis_pending[ids] = False


term._resample_command = _fixed.__get__(term)


def run(policy):
  torch.manual_seed(0)
  env.reset()
  terrain.terrain_types[:] = col
  terrain.terrain_levels[:] = 0
  terrain.env_origins[:] = terrain.terrain_origins[0, col]
  o = terrain.env_origins.clone()
  st = robot.data.default_root_state.clone()
  st[:, 0] = o[:, 0] + RING - start
  st[:, 1] = o[:, 1]
  st[:, 2] = o[:, 2] + 0.42
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
  term._resample_command(ids_all)
  obs, *_ = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=D))
  passed = torch.zeros(N, dtype=torch.bool, device=D)
  fell = torch.zeros_like(passed)
  t_pass = torch.full((N,), float("nan"), device=D)
  with torch.no_grad():
    for k in range(int(T / env.step_dt)):
      obs, *_ = env.step(policy(obs["actor"]))
      p = robot.data.root_link_pos_w
      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      fell |= tilt > math.radians(80)
      new = (
        (p[:, 0] - o[:, 0] > RING + WIDTH + 0.5)
        & (p[:, 2] - o[:, 2] < 0.55)
        & ~fell
        & ~passed
      )
      t_pass[new] = (k + 1) * env.step_dt
      passed |= new
  return passed, fell, t_pass


for spec in sys.argv[2:]:
  g, it = spec.split("@")
  ck = sorted(Path("logs/rsl_rl").glob(f"*/{g}"))[-1] / f"model_{it}.pt"
  passed, fell, t_pass = run(gb._load_policy(ck))
  n_fall = int(fell.sum())
  ub = 1 - 0.05 ** (1 / N) if n_fall == 0 else float("nan")
  stuck = ~passed & ~fell
  print(
    f"\n{ck.parent.name} @ {it}  {task}\n  bar {H:.2f} x {WIDTH:.2f} m, N {N} random draws"
    f" (vx 0.3-0.8, yaw +-20 deg, run-up 0.6-2.0 m, friction 0.4-1.0, obs noise on), T {T}s"
  )
  print(
    f"  pass {100 * passed.float().mean():.1f}%   fall {n_fall}/{N}"
    + (f" (95% upper bound {100 * ub:.1f}%)" if n_fall == 0 else "")
    + f"   stuck {int(stuck.sum())}/{N}   t_pass median {t_pass[passed].median().item():.1f} s"
  )
  for name, x, lo, hi in (
    ("vx", vx, 0.3, 0.8),
    ("friction", fric, 0.4, 1.0),
    ("run-up", start, 0.6, 2.0),
  ):
    edges = torch.linspace(lo, hi, 4, device=D)
    cells = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
      m = (x >= a) & (x <= b)
      cells.append(
        f"{a:.2f}-{b:.2f}: pass {100 * passed[m].float().mean():.0f}% fall {int(fell[m].sum())}"
      )
    print(f"  by {name:8s} " + " | ".join(cells))

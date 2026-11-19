"""Long-bar crossing benchmark (the 40 cm x 40 cm obstacle) for Go2W Climb policies.

Target (11-10): get over a long bar 40 cm high and 40 cm wide without falling;
standing on top is not required. The climbing column of the task's terrain is
replaced by climb_terrains.BoxBarRingTerrainCfg with one exact height per row
and a fixed width; the column keeps the name "high_platform", so the climb bit
and the climbing command treat it as a climb (as an operator would switch climb
mode on in front of the bar).

Every env starts at its tile centre facing +x (yaw +-YAW_DEG), 2 m from the
bar's near face, command vx = VX with heading 0, for T seconds.

Per height:
  cross%   base came down on the far side: more than 0.5 m past the bar's far
           face (x) and lower than 0.55 m (on the floor), not fallen
  fall%    tilted past 80 deg at any time
  top%     still up on the bar at the end (base z > h + 0.2)
  t_cross  median time of the crossing (s)
  then, over the crossings: share that touched with the trunk / thighs /
  shanks, p95 peak trunk force, and max pitch (deg).

usage: python tools/bar_bench.py <task> <run_glob>@<iter> [more specs ...]
env:   E (16)  HEIGHTS ("0.30,0.35,0.40,0.45,0.50")  WIDTH (0.40)  VX (0.5)
       T (15)  YAW_DEG (5)  NOISE (0)  FRIC (wheel friction, default the startup draw)
       START (2.0: distance from the start to the bar's near face, m)
       HEADING_DEG (0: commanded heading, i.e. crossing angle off square)
       STOPMID (0; 1: command 0 for 3 s once the base is over the bar's near face,
                then VX again -- the operator letting go of the stick mid-crossing)
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
E = int(os.environ.get("E", "16"))
HEIGHTS = tuple(
  float(x) for x in os.environ.get("HEIGHTS", "0.30,0.35,0.40,0.45,0.50").split(",")
)
WIDTH = float(os.environ.get("WIDTH", "0.40"))
VX = float(os.environ.get("VX", "0.5"))
T = float(os.environ.get("T", "15"))
YAW = math.radians(float(os.environ.get("YAW_DEG", "5")))
NOISE = os.environ.get("NOISE", "0") == "1"
FRIC = os.environ.get("FRIC")
START = float(os.environ.get("START", "2.0"))
HEADING = math.radians(float(os.environ.get("HEADING_DEG", "0")))
STOPMID = os.environ.get("STOPMID", "0") == "1"
CLIMB = "high_platform"
RING = 2.0

task = sys.argv[1]
cfg = load_env_cfg(task, play=True)
gen = cfg.scene.terrain.terrain_generator
flat = gen.sub_terrains["flat"]
gen = replace(
  gen,
  num_rows=len(HEIGHTS),
  num_cols=2,
  sub_terrains={
    "flat": replace(flat, proportion=0.5),
    CLIMB: BoxBarRingTerrainCfg(
      proportion=0.5,
      heights=HEIGHTS,
      bar_width_range=(WIDTH, WIDTH),
      ring_distance=RING,
    ),
  },
)
cfg.scene.terrain.terrain_generator = gen
cfg.scene.num_envs = len(HEIGHTS) * E
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.observations["actor"].enable_corruption = NOISE
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
terrain = env.scene.terrain
robot = env.scene["robot"]
col = list(gen.sub_terrains).index(CLIMB)
row_of = torch.arange(N, device=D) // E
h_of = torch.tensor(HEIGHTS, device=D)[row_of]
ids_all = torch.arange(N, device=D)
if FRIC:
  wl = torch.tensor(robot.find_geoms(".*_wheel_collision")[0], device=D)
  gid = robot.indexing.geom_ids.to(D)[wl].long()
  env.sim.model.geom_friction[:, gid, 0] = float(FRIC)

term = env.command_manager.get_term("twist")


def _fixed(self, ids):
  self.vel_command_b[ids, 0] = VX
  self.vel_command_b[ids, 1:] = 0.0
  self.heading_target[ids] = HEADING
  self.is_heading_env[ids] = True
  self.is_standing_env[ids] = False
  self.is_world_env[ids] = False
  self.is_forward_env[ids] = False
  if hasattr(self, "_climb"):
    self._climb[ids] = True
    self._axis_pending[ids] = False


term._resample_command = _fixed.__get__(term)
touch = {k: env.scene[f"{k}_ground_touch"] for k in ("trunk", "thigh", "shank")}


def run(policy):
  torch.manual_seed(0)
  env.reset()
  terrain.terrain_types[:] = col
  terrain.terrain_levels[:] = row_of
  terrain.env_origins[:] = terrain.terrain_origins[row_of, col]
  o = terrain.env_origins.clone()
  st = robot.data.default_root_state.clone()
  yaw = HEADING + (torch.rand(N, device=D) * 2 - 1) * YAW
  st[:, :3] = o + torch.tensor([RING - START, 0.0, 0.42], device=D)
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

  cross = torch.zeros(N, dtype=torch.bool, device=D)
  fell = torch.zeros_like(cross)
  t_cross = torch.full((N,), float("nan"), device=D)
  tch = {k: torch.zeros_like(cross) for k in touch}
  f_trunk = torch.zeros(N, device=D)
  pitch_max = torch.zeros(N, device=D)
  far = RING + WIDTH + 0.5
  stop_k = torch.full((N,), -1, dtype=torch.long, device=D)
  with torch.no_grad():
    for k in range(int(T / env.step_dt)):
      obs, *_ = env.step(policy(obs["actor"]))
      p = robot.data.root_link_pos_w
      x = p[:, 0] - o[:, 0]
      z = p[:, 2] - o[:, 2]
      g = robot.data.projected_gravity_b
      tilt = torch.acos((-g[:, 2]).clamp(-1, 1))
      fell |= tilt > math.radians(80)
      new = (x > far) & (z < 0.55) & ~fell & ~cross
      t_cross[new] = (k + 1) * env.step_dt
      active = ~cross & ~fell
      for key, s in touch.items():
        tch[key] |= (s.data.force_history.norm(dim=-1) > 1.0).flatten(1).any(1) & active
      ft = touch["trunk"].data.force_history.norm(dim=-1).flatten(1).max(1).values
      f_trunk = torch.where(active, torch.maximum(f_trunk, ft), f_trunk)
      pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2])).abs()
      pitch_max = torch.where(active, torch.maximum(pitch_max, pitch), pitch_max)
      cross |= new
      if STOPMID:
        go = (stop_k < 0) & (x > RING) & ~cross & ~fell
        if go.any():
          stop_k[go] = k
          term.vel_command_b[go, 0] = 0.0
        back = (stop_k >= 0) & (k - stop_k == 150)
        if back.any():
          term.vel_command_b[back, 0] = VX
  on_top = (z > h_of + 0.2) & ~fell
  return cross, fell, on_top, t_cross, tch, f_trunk, pitch_max


LOG_ROOT = Path("logs/rsl_rl")
for spec in sys.argv[2:]:
  g, it = spec.split("@")
  ck = sorted(LOG_ROOT.glob(f"*/{g}"))[-1] / f"model_{it}.pt"
  cross, fell, on_top, t_cross, tch, f_trunk, pitch_max = run(gb._load_policy(ck))
  print(
    f"\n{ck.parent.name} @ {it}   {task}   bar width {WIDTH} m  VX {VX}  T {T}s  E {E}"
    f"  noise {NOISE}  fric {FRIC or 'startup draw'}  start {START} m  heading {math.degrees(HEADING):.0f} deg"
    f"  yaw +-{math.degrees(YAW):.0f}  stopmid {STOPMID}"
  )
  print(
    "   h (m)  cross%  fall%  top%  t_cross | over crossings: touch% trunk thigh shank"
    "  F_trunk p95 N  pitch max p95"
  )
  for r, h in enumerate(HEIGHTS):
    s = row_of == r
    ok = cross & s
    n = ok.float().sum().clamp(min=1)
    tt = t_cross[ok]
    med = f"{tt.median().item():5.1f}" if len(tt) else "    -"
    f95 = f_trunk[ok].quantile(0.95).item() if ok.sum() > 1 else float("nan")
    p95 = pitch_max[ok].quantile(0.95).item() if ok.sum() > 1 else float("nan")
    print(
      f"   {h:.2f}  {100 * cross[s].float().mean():6.0f} {100 * fell[s].float().mean():6.0f}"
      f" {100 * on_top[s].float().mean():5.0f}   {med}  |"
      f"   {100 * (tch['trunk'] & ok).float().sum() / n:4.0f} {100 * (tch['thigh'] & ok).float().sum() / n:5.0f}"
      f" {100 * (tch['shank'] & ok).float().sum() / n:5.0f}   {f95:6.0f}   {p95:5.0f}",
      flush=True,
    )

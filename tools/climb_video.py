"""Record a Go2W Climb policy in front of a ledge of a given height (mp4).

Env 0 goes onto the climbing tile whose step height is closest to H, at the pit
centre facing +x, with vx = VX and heading 0; a side camera tracks the base.

usage: python tools/climb_video.py <task> <run_glob>@<iter> <out.mp4>
env:   H (0.5 m)  VX (0.5)  T (8 s)  AZ (90, camera azimuth)  ELEV (-10)  DIST (3.0)
       EXACT (1: every climbing tile at exactly H, instead of the row nearest H)
       BAR (1: the climbing column becomes the long-bar ring, H high, WIDTH (0.40) wide)
       FLAT (1: on the flat column instead, command CMD "vx,vy,wz" (default "0,0,0.5"),
             no heading control, climb bit 0 -- for posture clips)
"""

import contextlib
import io
import os
import sys
import types
from pathlib import Path

import mediapy
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D
H = float(os.environ.get("H", "0.5"))
VX = float(os.environ.get("VX", "0.5"))
T = float(os.environ.get("T", "8"))
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

task, spec, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 4
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
if os.environ.get("EXACT") == "1":
  cfg.scene.terrain.terrain_generator.sub_terrains[
    "high_platform"
  ].step_height_range = (H, H)
BAR = os.environ.get("BAR") == "1"
if BAR:
  from dataclasses import replace as _replace

  from mjlab_playground.velocity.config.go2w_climb.climb_terrains import (
    BoxBarRingTerrainCfg,
  )

  _w = float(os.environ.get("WIDTH", "0.40"))
  _gen = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = _replace(
    _gen,
    num_rows=1,
    num_cols=2,
    sub_terrains={
      "flat": _replace(_gen.sub_terrains["flat"], proportion=0.5),
      "high_platform": BoxBarRingTerrainCfg(
        proportion=0.5, heights=(H,), bar_width_range=(_w, _w), ring_distance=2.0
      ),
    },
  )
cfg.viewer.env_idx = 0
cfg.viewer.max_extra_envs = 0
cfg.viewer.distance = float(os.environ.get("DIST", "3.0"))
cfg.viewer.elevation = float(os.environ.get("ELEV", "-10"))
cfg.viewer.azimuth = float(os.environ.get("AZ", "90"))
cfg.viewer.height, cfg.viewer.width = 480, 640
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D, render_mode="rgb_array")
terrain = env.scene.terrain
robot = env.scene["robot"]
col = list(cfg.scene.terrain.terrain_generator.sub_terrains).index("high_platform")
h_rows = (
  torch.full((terrain.terrain_origins.shape[0],), H, device=D)
  if BAR
  else -terrain.terrain_origins[:, col, 2] / 2
)
row = int(torch.argmin((h_rows - H).abs()))

term = env.command_manager.get_term("twist")
FLAT = os.environ.get("FLAT") == "1"
CMD = [float(x) for x in os.environ.get("CMD", "0,0,0.5").split(",")]
if FLAT:
  col = list(cfg.scene.terrain.terrain_generator.sub_terrains).index("flat")
  row = 0


def _fixed(self, ids):
  if FLAT:
    self.vel_command_b[ids] = torch.tensor(CMD, device=self.device)
    self.is_heading_env[ids] = False
    self.is_standing_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False
    if hasattr(self, "_climb"):
      self._climb[ids] = False
    return
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

g, it = spec.split("@")
ck = sorted((Path("logs/rsl_rl") / LOGS[task]).glob(g))[-1] / f"model_{it}.pt"
policy = gb._load_policy(ck)

env.reset()
N = env.num_envs
terrain.terrain_types[:] = col
terrain.terrain_levels[:] = row
terrain.env_origins[:] = terrain.terrain_origins[row, col]
st = robot.data.default_root_state.clone()
st[:, :3] = terrain.env_origins + torch.tensor([0.0, 0.0, 0.42], device=D)
st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=D)
st[:, 7:] = 0.0
robot.write_root_state_to_sim(st)
robot.write_joint_state_to_sim(
  robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
)
term._resample_command(torch.arange(N, device=D))
obs, *_ = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=D))

frames = []
with torch.no_grad():
  for _ in range(int(T / env.step_dt)):
    obs, *_ = env.step(policy(obs["actor"]))
    frames.append(env.render())
z = (robot.data.root_link_pos_w[0, 2] - terrain.env_origins[0, 2]).item()
mediapy.write_video(out, frames, fps=int(round(1 / env.step_dt)))
print(
  f"{ck.parent.name}@{it}: row {row} h {h_rows[row].item():.3f} m, final base z rel. floor {z:.2f}, -> {out}"
)

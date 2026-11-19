# Flat-ground tracking of a Climb policy: all envs on the "flat" column, fixed commands.
import contextlib
import io
import sys
import types

import torch

sys.path.insert(0, "tools")
from pathlib import Path

import gait_bench as gb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab_playground.velocity import mdp

D = "cuda:0"
gb.DEVICE = D
task, spec, logdir = sys.argv[1], sys.argv[2], sys.argv[3]
CMDS = [
  ("stand", (0, 0, 0)),
  ("vx0.3", (0.3, 0, 0)),
  ("vx1.0", (1.0, 0, 0)),
  ("vx-0.5", (-0.5, 0, 0)),
  ("vy0.3", (0, 0.3, 0)),
  ("wz0.5", (0, 0, 0.5)),
  ("wz1.0", (0, 0, 1.0)),
  ("combo", (0.5, 0.2, 0.5)),
]
E = 32
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = E * len(CMDS)
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
terrain, robot = env.scene.terrain, env.scene["robot"]
table = torch.tensor(
  [c for _, c in CMDS for _ in range(E)], dtype=torch.float32, device=D
)
term = env.command_manager.get_term("twist")


def fixed(self, ids):
  self.vel_command_b[ids] = table[ids]
  self.is_heading_env[ids] = False
  self.is_standing_env[ids] = False
  self.is_world_env[ids] = False
  self.is_forward_env[ids] = False


term._resample_command = types.MethodType(fixed, term)
env.reset()
terrain.terrain_types[:] = 0
terrain.terrain_levels[:] = 0
terrain.env_origins[:] = terrain.terrain_origins[0, 0]
st = robot.data.default_root_state.clone()
st[:, :3] = terrain.env_origins + torch.tensor([0, 0, 0.42], device=D)
st[:, 7:] = 0
robot.write_root_state_to_sim(st)
robot.write_joint_state_to_sim(
  robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
)
term._resample_command(torch.arange(N, device=D))
obs, *_ = env.step(torch.zeros(N, 16, device=D))
g, it = spec.split("@")
ck = sorted(Path(f"logs/rsl_rl/{logdir}").glob(g))[-1] / f"model_{it}.pt"
pol = gb._load_policy(ck)
wid, _ = robot.find_joints(".*_wheel_joint")
wid = torch.tensor(wid, device=D)
wabs = torch.zeros(N, device=D)
wsig = torch.zeros(N, device=D)
contact = env.scene["feet_ground_contact"]
air = torch.zeros(N, device=D)
err_v = torch.zeros(N, device=D)
err_w = torch.zeros(N, device=D)
n = 0
vel_xy = torch.zeros(N, 2, device=D)
wsum = torch.zeros(N, device=D)
p0 = None
with torch.no_grad():
  for k in range(500):
    obs, *_ = env.step(pol(obs["actor"]))
    if k == 150:
      p0 = robot.data.root_link_pos_w[:, :2].clone()
    if k >= 150:
      v = quat_apply_inverse(
        yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w
      )[:, :2]
      w = robot.data.root_link_ang_vel_w[:, 2]
      vel_xy += v
      wsum += w
      n += 1
      wv = robot.data.joint_vel[:, wid]
      wabs += wv.abs().mean(1)
      wsig += wv.mean(1)
      air += (contact.data.current_air_time > 0.02).float().sum(1)
p1 = robot.data.root_link_pos_w[:, :2]
print(
  f"{ck.parent.name}@{it} on flat ground (climb bit = 0), 32 envs/cmd, 7 s after 3 s settle"
)
print("  cmd       target (vx,vy,wz)    actual vx    vy     wz    | stand drift cm")
for i, (name, c) in enumerate(CMDS):
  s = slice(i * E, (i + 1) * E)
  av = (vel_xy[s] / n).mean(0)
  aw = (wsum[s] / n).mean()
  drift = (p1[s] - p0[s]).norm(dim=1).mean() * 100
  extra = (
    f"| drift {drift:5.1f} cm, |wheel| {(wabs[s] / n).mean():.2f} rad/s, signed {(wsig[s] / n).mean():.2f}, feet in air {(air[s] / n).mean():.2f}"
    if name == "stand"
    else ""
  )
  print(f"  {name:7s}  {str(c):18s}  {av[0]:6.2f} {av[1]:6.2f} {aw:6.2f}  " + extra)

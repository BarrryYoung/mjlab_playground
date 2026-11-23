# Startup heading drift on a plane, platform-style: spawn at rest in the default
# pose, step command (vx, 0, 0) from the first control step, open loop for 14 s.
# Reports heading change vs spawn at 0.5/1/2/3/14 s and the max cross-track offset.
# usage: python tools/startup_yaw_bench.py CKPT [--task T] [--base-mass KG] [--vx 0.6]
import argparse
import contextlib
import io
import math
import sys
import types

import torch

sys.path.insert(0, "tools")
from pathlib import Path

import gait_bench as gb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab_playground.velocity import mdp

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--task", default="Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Unitree-Go2W")
ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--vx", type=float, default=0.6)
ap.add_argument("--base-mass", type=float, default=None, help="override base_link mass (kg)")
ap.add_argument("--spawn-z", type=float, default=0.42)
ap.add_argument("--pre", choices=("none", "fixstand", "policystand"), default="none",
                help="before the step command: fixstand = hold the default pose with zero "
                "actions (PD only, like the deploy FixStand state); policystand = policy with cmd 0")
ap.add_argument("--pre-s", type=float, default=2.0)
args = ap.parse_args()

D = "cuda:0"
gb.DEVICE = D
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.envs
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
for k in ("push_robot",):
  cfg.events.pop(k, None)
if "base_mass" in cfg.events:  # keep the field per-world, no random offset
  cfg.events["base_mass"].params["ranges"] = (0.0, 0.0)
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
cmd = torch.tensor([args.vx, 0.0, 0.0], device=D)


def fixed(self, ids):
  self.vel_command_b[ids] = cmd
  for name in dir(self):
    v = getattr(self, name, None)
    if (name.startswith("is_") or name in ("straight", "heading_env")) and isinstance(
      v, torch.Tensor
    ) and v.dtype == torch.bool:
      v[ids] = False


def no_update(self):
  self.vel_command_b[:] = cmd


term._resample_command = types.MethodType(fixed, term)
term._update_command = types.MethodType(no_update, term)
env.reset()
bid = robot.find_bodies("base_link")[0][0]
gbid = robot.indexing.body_ids[bid] if hasattr(robot, "indexing") else None
if args.base_mass is not None:
  m = env.sim.model.body_mass
  col = gbid if gbid is not None else bid + 1
  m[:, col] = args.base_mass
print("base_link mass now:", float(env.sim.model.body_mass[0, gbid if gbid is not None else bid + 1]))

st = robot.data.default_root_state.clone()
st[:, :3] = env.scene.env_origins + torch.tensor([0, 0, args.spawn_z], device=D)
st[:, 7:] = 0
robot.write_root_state_to_sim(st)
robot.write_joint_state_to_sim(
  robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
)
term._resample_command(torch.arange(N, device=D))
obs, *_ = env.step(torch.zeros(N, 16, device=D))
pol = gb._load_policy(Path(args.ckpt))

if args.pre != "none":
  cmd.zero_()
  term.vel_command_b[:] = 0
  with torch.no_grad():
    for _ in range(int(args.pre_s / env.step_dt)):
      a = torch.zeros(N, 16, device=D) if args.pre == "fixstand" else pol(obs["actor"])
      obs, *_ = env.step(a)
  cmd[0] = args.vx
  term.vel_command_b[:] = cmd
  obs = env.observation_manager.compute()
  print(f"pre-phase {args.pre} {args.pre_s} s done; base z {robot.data.root_link_pos_w[:, 2].mean():.3f}, "
        f"|v| {robot.data.root_link_lin_vel_w[:, :2].norm(dim=1).mean():.3f} m/s")


def yaw():
  q = robot.data.root_link_quat_w
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


y0 = yaw().clone()
p0 = robot.data.root_link_pos_w[:, :2].clone()
dir0 = torch.stack([torch.cos(y0), torch.sin(y0)], -1)
marks = {25: "0.5s", 50: "1s", 100: "2s", 150: "3s", 700: "14s"}
out = {}
xt_max = torch.zeros(N, device=D)
with torch.no_grad():
  for k in range(1, 701):
    obs, *_ = env.step(pol(obs["actor"]))
    d = robot.data.root_link_pos_w[:, :2] - p0
    xt = (d[:, 0] * dir0[:, 1] - d[:, 1] * dir0[:, 0]).abs()
    xt_max = torch.maximum(xt_max, xt)
    if k in marks:
      dy = torch.remainder(yaw() - y0 + math.pi, 2 * math.pi) - math.pi
      out[marks[k]] = torch.rad2deg(dy)
      if k == 150:
        xt3 = xt_max.clone()
prog = (d * dir0).sum(-1)
print(f"{Path(args.ckpt).parent.name}/{Path(args.ckpt).name}  vx={args.vx}  envs={N}  pre={args.pre}")
print("heading change vs spawn (deg, + = left): median [p10, p90]")
for k, v in out.items():
  q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], device=D))
  print(f"  {k:>5s}: {q[1]:+6.1f}  [{q[0]:+6.1f}, {q[2]:+6.1f}]")
print(f"max cross-track: first 3 s median {xt3.median():.2f} m; 14 s median {xt_max.median():.2f} m, "
      f"frac <= 0.45 m: {(xt_max <= 0.45).float().mean():.2f}; progress median {prog.median():.1f} m")

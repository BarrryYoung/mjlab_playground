# Zero-command creep on a plane (11-24, for the WheelPos task).
# Scenarios:
#   stand: spawn at rest, policy with cmd 0 for T s.
#   stop:  cmd (vx, 0, 0) for --drive-s, then cmd 0 for T s (braking, then holding).
# After the command goes to 0: base displacement at 1/5/10/T s, mean base speed and
# mean |wheel speed| over the last 10 s, heading change. Actor noise on by default
# (the robot sees noisy readings); --no-noise turns it off.
# usage: python tools/zero_cmd_bench.py CKPT --task T [--scenario stand|stop] [--no-noise]
import argparse
import contextlib
import io
import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, "tools")
import gait_bench as gb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg

import mjlab_playground  # noqa: F401
from mjlab_playground.velocity import mdp

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--task", required=True)
ap.add_argument("--envs", type=int, default=128)
ap.add_argument("--scenario", choices=("stand", "stop"), default="stand")
ap.add_argument("--vx", type=float, default=1.0)
ap.add_argument("--drive-s", type=float, default=3.0)
ap.add_argument("--hold-s", type=float, default=20.0)
ap.add_argument("--no-noise", action="store_true")
ap.add_argument("--wheel-deadband", type=float, default=0.0,
                help="部署端滤波模拟: 指令为 0 时 |目标轮速|(rad/s) 小于它的轮子置 0")
ap.add_argument("--wheel-fade-tau", type=float, default=0.0,
                help="部署端渐变刹车: 指令为 0 且估算速度 < --fade-gate 后，轮子目标乘 g，g 按时间常数 tau(s) 指数衰减到 0；有指令立刻 g=1")
ap.add_argument("--fade-gate", type=float, default=0.05,
                help="门控速度 (m/s)，按 4 轮 |轮速| 均值 x 轮半径估算（真机可用）")
args = ap.parse_args()

D = "cuda:0"
gb.DEVICE = D
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.envs
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.events.pop("init_rough_terrain_at_top", None)
cfg.observations["actor"].enable_corruption = not args.no_noise
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
cmd = torch.zeros(3, device=D)


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
obs, _ = env.reset()
_pol = gb._load_policy(Path(args.ckpt))
_wheel_act = slice(12, 16)
_wheel_scale = 5.0  # wheel_vel action scale


_WHEEL_R = 0.086
_fade = {"g": None, "armed": None}


def pol(o):
  a = _pol(o)
  zero = bool((cmd == 0).all())
  if args.wheel_deadband > 0 and zero:
    w = a[:, _wheel_act]
    a[:, _wheel_act] = torch.where(w.abs() * _wheel_scale < args.wheel_deadband, 0.0, w)
  if args.wheel_fade_tau > 0:
    if _fade["g"] is None:
      _fade["g"] = torch.ones(a.shape[0], device=a.device)
      _fade["armed"] = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
    if not zero:
      _fade["g"][:] = 1.0
      _fade["armed"][:] = False
    else:
      est = robot.data.joint_vel[:, wid].abs().mean(dim=1) * _WHEEL_R
      _fade["armed"] |= est < args.fade_gate
      decay = math.exp(-env.step_dt / args.wheel_fade_tau)
      _fade["g"] = torch.where(_fade["armed"], _fade["g"] * decay, _fade["g"])
    a[:, _wheel_act] = a[:, _wheel_act] * _fade["g"].unsqueeze(-1)
  return a
wid = robot.find_joints(
  ["FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"], preserve_order=True
)[0]


def yaw():
  w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


with torch.no_grad():
  # settle 1 s at cmd 0 in both scenarios
  for _ in range(int(1.0 / env.step_dt)):
    obs, *_ = env.step(pol(obs["actor"]))
  if args.scenario == "stop":
    cmd[0] = args.vx
    term.vel_command_b[:] = cmd
    for _ in range(int(args.drive_s / env.step_dt)):
      obs, *_ = env.step(pol(obs["actor"]))
    cmd.zero_()
    term.vel_command_b[:] = 0
  p0 = robot.data.root_link_pos_w[:, :2].clone()
  y0 = yaw().clone()
  steps = int(args.hold_s / env.step_dt)
  marks = {int(s / env.step_dt): f"{s:g}s" for s in (1, 5, 10, args.hold_s)}
  disp = {}
  last = int(10.0 / env.step_dt)
  sp_sum = torch.zeros(N, device=D)
  wv_sum = torch.zeros(N, device=D)
  for k in range(1, steps + 1):
    obs, *_ = env.step(pol(obs["actor"]))
    if k > steps - last:
      sp_sum += robot.data.root_link_lin_vel_w[:, :2].norm(dim=1)
      wv_sum += robot.data.joint_vel[:, wid].abs().mean(dim=1)
    if k in marks:
      disp[marks[k]] = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)
  dyaw = torch.rad2deg(torch.remainder(yaw() - y0 + math.pi, 2 * math.pi) - math.pi).abs()
  up = robot.data.projected_gravity_b[:, 2] < -0.8

name = f"{Path(args.ckpt).parent.name[:40]}/{Path(args.ckpt).name}"
print(f"{name}  scenario={args.scenario}  noise={'off' if args.no_noise else 'on'}  envs={N}  upright {up.float().mean():.2f}")
q = torch.tensor([0.5, 0.9], device=D)
print("  displacement since cmd 0 (cm): median / p90")
for k, v in disp.items():
  m, p = torch.quantile(v * 100, q).tolist()
  print(f"    {k:>4s}: {m:6.1f} / {p:6.1f}")
n = min(last, steps)
print(f"  last {n * env.step_dt:.0f} s: base speed {(sp_sum / n).median() * 100:.2f} cm/s (median), "
      f"|wheel vel| {(wv_sum / n).median():.3f} rad/s (median), heading change {dyaw.median():.1f} deg")

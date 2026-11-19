"""Posture / gait comparison on flat ground, for any Go2W task (flat or Climb).

Puts every env on a plane (Climb tasks: the terrain generator is swapped for a
plane, so the climb bit reads 0), pins a fixed command per group of 32 envs,
runs the policy deterministically, drops 2 s and measures 6 s. The numbers are
read off the robot itself, so policies with different observation layouts
compare directly.

Per command:
  track      actual vx vy wz (yaw frame)
  z          base height above the ground (m)
  roll/pitch mean |angle| (deg)
  dq hip/thigh/calf   mean |q - q_default| per joint group (rad)
  da         RMS step-to-step change of the policy output (jerkiness)
  ddq        RMS leg joint acceleration (rad/s^2, /100)
  tau        RMS leg torque (N m); zero command: mean |hip torque| too
  air        share of wheel-time off the ground; lifts per wheel per second
  zero cmd:  mean |wheel speed| (rad/s) and xy drift over the 6 s (cm)

usage: python tools/posture_bench.py <task> <log_dir_glob>@<iter> [<task> <glob>@<iter> ...]
       globs resolve under logs/rsl_rl/*/
"""

import contextlib
import io
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
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D
CMDS = [
  ("zero", (0.0, 0.0, 0.0)),
  ("vx0.5", (0.5, 0.0, 0.0)),
  ("vx1.0", (1.0, 0.0, 0.0)),
  ("vy0.25", (0.0, 0.25, 0.0)),
  ("wz0.15", (0.0, 0.0, 0.15)),
  ("wz0.5", (0.0, 0.0, 0.5)),
  ("wz1.0", (0.0, 0.0, 1.0)),
]
E = 32
WARM = 100  # policy steps (50 Hz): 2 s
MEAS = int(50 * float(os.environ.get("MEAS_S", "6")))  # measured window, default 6 s
# SPLAY=1: also print each leg's signed hip angle (deg, + = foot outward, − = inward)
# over the first and last third of the window, to see legs splaying as a walk goes on.
SPLAY = os.environ.get("SPLAY", "0") == "1"


def make_env(task: str) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(task, play=True)
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "generator":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
  cfg.scene.num_envs = E * len(CMDS)
  cfg.episode_length_s = 1e6
  cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  with contextlib.redirect_stdout(io.StringIO()):
    return ManagerBasedRlEnv(cfg, device=D)


def bench(task: str, spec: str) -> None:
  g, it = spec.split("@")
  hits = sorted(Path("logs/rsl_rl").glob(f"*/{g}"))
  assert hits, f"no run matches {g}"
  ck = hits[-1] / f"model_{it}.pt"
  env = make_env(task)
  N = env.num_envs
  robot = env.scene["robot"]
  table = torch.tensor(
    [c for _, c in CMDS for _ in range(E)], dtype=torch.float32, device=D
  )
  term = env.command_manager.get_term("twist")

  def fixed(self, ids):
    self.vel_command_b[ids] = table[ids]
    self.is_standing_env[ids] = False
    self.is_heading_env[ids] = False
    for flag in ("is_world_env", "is_forward_env", "is_low_env"):
      if hasattr(self, flag):
        getattr(self, flag)[ids] = False

  term._resample_command = types.MethodType(fixed, term)
  obs, _ = env.reset()
  term._resample_command(torch.arange(N, device=D))
  pol = gb._load_policy(ck)

  jn = robot.joint_names
  leg = torch.tensor([i for i, n in enumerate(jn) if "wheel" not in n], device=D)
  grp = {
    k: torch.tensor([i for i, n in enumerate(jn) if k in n], device=D)
    for k in ("hip", "thigh", "calf")
  }
  wheel = torch.tensor([i for i, n in enumerate(jn) if "wheel" in n], device=D)
  q0 = robot.data.default_joint_pos
  contact = env.scene["feet_ground_contact"]

  acc = {
    k: torch.zeros(N, device=D)
    for k in (
      "vx",
      "vy",
      "wz",
      "z",
      "roll",
      "pitch",
      "dhip",
      "dthigh",
      "dcalf",
      "da2",
      "ddq2",
      "tau2",
      "hip_tau",
      "air",
      "lifts",
      "wabs",
    )
  }
  # Hip axis is +x: +q swings the foot toward +y, i.e. outward on the left legs.
  hip_ids = [jn.index(f"{leg}_hip_joint") for leg in ("FR", "FL", "RR", "RL")]
  side = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=D)
  third = MEAS // 3
  hip_early = torch.zeros(N, 4, device=D)
  hip_late = torch.zeros(N, 4, device=D)
  prev_a = None
  prev_found = None
  p0 = None
  with torch.no_grad():
    for k in range(WARM + MEAS):
      a = pol(obs["actor"])
      obs, *_ = env.step(a)
      if k < WARM:
        prev_a = a
        continue
      if p0 is None:
        p0 = robot.data.root_link_pos_w[:, :2].clone()
      v = quat_apply_inverse(
        yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w
      )
      acc["vx"] += v[:, 0]
      acc["vy"] += v[:, 1]
      acc["wz"] += robot.data.root_link_ang_vel_w[:, 2]
      acc["z"] += robot.data.root_link_pos_w[:, 2]
      gvec = robot.data.projected_gravity_b
      acc["roll"] += torch.rad2deg(torch.atan2(gvec[:, 1], -gvec[:, 2])).abs()
      acc["pitch"] += torch.rad2deg(torch.atan2(-gvec[:, 0], -gvec[:, 2])).abs()
      hip_out = torch.rad2deg((robot.data.joint_pos - q0)[:, hip_ids]) * side
      if k - WARM < third:
        hip_early += hip_out / third
      elif k - WARM >= MEAS - third:
        hip_late += hip_out / third
      dq = (robot.data.joint_pos - q0).abs()
      acc["dhip"] += dq[:, grp["hip"]].mean(1)
      acc["dthigh"] += dq[:, grp["thigh"]].mean(1)
      acc["dcalf"] += dq[:, grp["calf"]].mean(1)
      acc["da2"] += ((a - prev_a) ** 2).mean(1)
      prev_a = a
      acc["ddq2"] += (robot.data.joint_acc[:, leg] ** 2).mean(1)
      tau = robot.data.qfrc_actuator
      acc["tau2"] += (tau[:, leg] ** 2).mean(1)
      acc["hip_tau"] += tau[:, grp["hip"]].abs().mean(1)
      found = contact.data.found > 0  # [B, 4]
      acc["air"] += (~found).float().mean(1)
      if prev_found is not None:
        acc["lifts"] += (prev_found & ~found).float().sum(1)
      prev_found = found
      acc["wabs"] += robot.data.joint_vel[:, wheel].abs().mean(1)
  drift = (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=1)
  m = {k: v / MEAS for k, v in acc.items()}
  m["lifts"] = acc["lifts"] / (4 * MEAS * env.step_dt)  # per wheel per second

  print(f"\n{task}  {ck.parent.name[-60:]} @ {it}")
  print(
    "  cmd      vx    vy    wz  |   z   roll pitch | dq hip thigh calf |   da  ddq/100  tau"
    "  | air%  lifts/s"
  )
  for i, (name, _c) in enumerate(CMDS):
    s = slice(i * E, (i + 1) * E)
    f = {k: v[s].mean().item() for k, v in m.items()}
    line = (
      f"  {name:6s} {f['vx']:5.2f} {f['vy']:5.2f} {f['wz']:5.2f} | {f['z']:.3f} {f['roll']:5.1f}"
      f" {f['pitch']:5.1f} |  {f['dhip']:.3f} {f['dthigh']:.3f} {f['dcalf']:.3f} |"
      f" {f['da2'] ** 0.5:.3f}  {f['ddq2'] ** 0.5 / 100:6.2f}  {f['tau2'] ** 0.5:4.1f}"
      f"  | {100 * f['air']:4.1f}  {f['lifts']:5.2f}"
    )
    if name == "zero":
      line += (
        f"   wheel {f['wabs']:.2f} rad/s  drift {100 * drift[s].mean():.1f} cm"
        f"  |hip tau| {f['hip_tau']:.1f}"
      )
    print(line, flush=True)
  if SPLAY:
    print(
      f"  hip out+ deg  FR FL RR RL   first {third / 50:.0f} s  ->  last {third / 50:.0f} s"
    )
    for i, (name, _) in enumerate(CMDS):
      s_ = slice(i * E, (i + 1) * E)
      e = " ".join(f"{x:5.1f}" for x in hip_early[s_].mean(0).tolist())
      l_ = " ".join(f"{x:5.1f}" for x in hip_late[s_].mean(0).tolist())
      print(f"  {name:6s}  {e}   ->  {l_}", flush=True)
  env.close()


args = sys.argv[1:]
for t, sp in zip(args[0::2], args[1::2], strict=True):
  bench(t, sp)

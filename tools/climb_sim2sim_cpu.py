"""Run a Climb policy in classic CPU MuJoCo: a sim2sim check against mujoco_warp.

The task env is built once with num_envs=1, only to get its compiled MjModel
(terrain + robot, MujocoCfg applied) and the terrain origins. After that it is
plain mujoco.mj_step on the CPU: the actor observation (robot_lab 57 + the
187-ray height scan, via mj_ray) and the PD / wheel targets are recomputed here
from the task definition, independently of mjlab's managers.

Blind tasks: per-frame terms (robot_lab 57 + climb-mode bit = 1) with 6 frames
of history laid out as mjlab does it -- per term, oldest -> newest, flattened,
then the terms concatenated; the first frame after a reset back-fills the
history.

DC tasks: the legs are <motor> actuators in the compiled model, so the PD
torque (kp 25, kd 0.5, nominal) is computed here every physics step and
clipped to the DC torque-speed curve as mjlab's DcMotorActuator does
(hip / thigh 23.5 N m at rest -> 0 at 30 rad/s, calf 35.55 N m -> 0 at 20 rad/s);
no command delay.

The robot starts at the pit centre facing +x on the climbing tile whose step is
closest to each requested height; command vx = VX, heading 0 (yaw rate from the
same heading controller: 0.5 * error, clipped to +-1). Success = both rises
(Chebyshev distance > 3.15 m and base z > 2h + 0.15 m above the pit floor);
fall = tilt > 80 deg before that.

BAR=1: the climbing column becomes the long-bar ring (climb_terrains.
BoxBarRingTerrainCfg, one row per H, width WIDTH, inner face 2 m from the
start); success = more than 0.5 m past the bar's far face and back on the
floor (base z < 0.55 m); fall = tilt > 80 deg at any time.

usage: python tools/climb_sim2sim_cpu.py <task> <run_glob>@<iter>
env:   H ("0.3,0.4,0.45,0.5")  VX (0.5)  T (12 s)  YAW_DEG (0, initial yaw)  OUT (mp4 of the last height, optional)
       BAR (0)  WIDTH (0.40)
"""

import contextlib
import io
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

gb.DEVICE = "cpu"
HS = [float(x) for x in os.environ.get("H", "0.3,0.4,0.45,0.5").split(",")]
VX = float(os.environ.get("VX", "0.5"))
T = float(os.environ.get("T", "12"))
YAW0 = math.radians(float(os.environ.get("YAW_DEG", "0")))
OUT = os.environ.get("OUT")
BAR = os.environ.get("BAR", "0") == "1"
WIDTH = float(os.environ.get("WIDTH", "0.40"))
VIEW = os.environ.get("VIEW", "0") == "1"  # live passive viewer, real time
LOGS = {
  "Playground-Velocity-Climb-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-H65-Unitree-Go2W": "playground_go2w_climb_velocity",
  "Playground-Velocity-Climb-Blind-Unitree-Go2W": "playground_go2w_climb_blind_velocity",
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
HIST = 6
LEGS = [
  f"{leg}_{j}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for j in ("hip", "thigh", "calf")
]
WHEELS = [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.5, "wheel": 0.0}
LEG_SCALE, WHEEL_SCALE = 0.25, 5.0
DECIMATION = 4

task, spec = sys.argv[1], sys.argv[2]
BLIND = "Blind" in task
DC = "-DC-" in task
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 1
if BAR:
  from dataclasses import replace as _replace

  from mjlab_playground.velocity.config.go2w_climb.climb_terrains import (
    BoxBarRingTerrainCfg,
  )

  _gen = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = _replace(
    _gen,
    num_rows=len(HS),
    num_cols=2,
    sub_terrains={
      "flat": _replace(_gen.sub_terrains["flat"], proportion=0.5),
      "high_platform": BoxBarRingTerrainCfg(
        proportion=0.5,
        heights=tuple(HS),
        bar_width_range=(WIDTH, WIDTH),
        ring_distance=2.0,
      ),
    },
  )
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
m = env.sim.mj_model
origins = env.scene.terrain.terrain_origins.cpu().numpy()
col = list(cfg.scene.terrain.terrain_generator.sub_terrains).index("high_platform")
h_rows = np.array(HS) if BAR else -origins[:, col, 2] / 2
assert abs(m.opt.timestep - 0.005) < 1e-9, m.opt.timestep


def _id(kind, name):
  i = mujoco.mj_name2id(m, kind, name)
  assert i >= 0, name
  return i


jid = {n: _id(mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}") for n in LEGS + WHEELS}
qadr = np.array([m.jnt_qposadr[jid[n]] for n in LEGS + WHEELS])
vadr = np.array([m.jnt_dofadr[jid[n]] for n in LEGS + WHEELS])
default = np.array([DEFAULT[n.split("_")[1]] for n in LEGS + WHEELS])
act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
act = np.array([act_of_joint[jid[n]] for n in LEGS + WHEELS])
base = _id(mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
free = _id(mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
fq, fv = m.jnt_qposadr[free], m.jnt_dofadr[free]
gyro = _id(mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
gadr = m.sensor_adr[gyro]

KP, KD = 25.0, 0.5
SAT = np.array([35.55 if "calf" in n else 23.5 for n in LEGS])
VMAX = np.array([20.0 if "calf" in n else 30.0 for n in LEGS])
if DC:
  assert all(m.actuator_biastype[a] == 0 for a in act[:12]), (
    "legs are not <motor> actuators"
  )


def dc_torque(d, q_target):
  q, qd = d.qpos[qadr[:12]], d.qvel[vadr[:12]]
  tau = KP * (q_target - q) - KD * qd
  vc = np.clip(qd, -2 * VMAX, 2 * VMAX)  # corner velocity, effort == saturation
  top = np.minimum(SAT * (1.0 - vc / VMAX), SAT)
  bot = np.maximum(SAT * (-1.0 - vc / VMAX), -SAT)
  return np.clip(tau, bot, top)


# Height scan: mjlab GridPatternCfg(1.6 x 1.0, 0.1), meshgrid "xy" -> y-major,
# x fastest; origins 1 m above the base in the yaw frame; terrain = group 0.
xs = np.arange(-0.8, 0.8 + 0.05, 0.1)
ys = np.arange(-0.5, 0.5 + 0.05, 0.1)
gx, gy = np.meshgrid(xs, ys, indexing="xy")
grid = np.stack([gx.ravel(), gy.ravel()], 1)
assert len(grid) == 187
geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
ray_geom = np.zeros(1, dtype=np.int32)
down = np.array([0.0, 0.0, -1.0])


def height_scan(d, yaw):
  p = d.xpos[base]
  c, s = math.cos(yaw), math.sin(yaw)
  out = np.empty(len(grid))
  for i, (ox, oy) in enumerate(grid):
    pnt = np.array([p[0] + c * ox - s * oy, p[1] + s * ox + c * oy, p[2] + 1.0])
    dist = mujoco.mj_ray(m, d, pnt, down, geomgroup, 1, base, ray_geom)
    out[i] = 5.0 if (dist < 0 or dist > 5.0) else dist - 1.0
  return out


def observe(d, last_action, hist):
  R = d.xmat[base].reshape(3, 3)
  fwd = R[:, 0]
  yaw = math.atan2(fwd[1], fwd[0])
  wz = float(np.clip(0.5 * math.atan2(math.sin(-yaw), math.cos(-yaw)), -1.0, 1.0))
  jp = d.qpos[qadr] - default
  jp[12:] = 0.0
  terms = [
    d.sensordata[gadr : gadr + 3] * 0.25,
    R.T @ np.array([0.0, 0.0, -1.0]),
    np.array([VX, 0.0, wz]),
    jp,
    d.qvel[vadr] * 0.05,
    last_action,
  ]
  if not BLIND:
    return np.concatenate(terms + [height_scan(d, yaw) / 5.0]).astype(np.float32), R
  if "NoBit" not in task:
    terms.append(np.array([1.0]))  # climb mode on
  if not hist:  # first frame back-fills the history
    hist.extend([[t.copy() for _ in range(HIST)] for t in terms])
  else:
    for buf, t in zip(hist, terms, strict=True):
      buf.pop(0)
      buf.append(t.copy())
  return np.concatenate([np.concatenate(buf) for buf in hist]).astype(np.float32), R


g, it = spec.split("@")
ck = sorted((Path("logs/rsl_rl") / LOGS[task]).glob(g))[-1] / f"model_{it}.pt"
policy = gb._load_policy(ck)
print(f"{ck.parent.name}@{it} in CPU MuJoCo {mujoco.__version__}; VX {VX}, T {T}s")

renderer = None
viewer = None
if VIEW:
  import time

  import mujoco.viewer

  vd = mujoco.MjData(m)
  viewer = mujoco.viewer.launch_passive(m, vd)
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  viewer.cam.trackbodyid = base
  viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.6, -25.0, 90.0
for k, H in enumerate(HS):
  row = int(np.argmin(np.abs(h_rows - H)))
  o = origins[row, col]
  h = h_rows[row]
  d = mujoco.MjData(m)
  d.qpos[fq : fq + 3] = o + np.array([0.0, 0.0, 0.42])
  d.qpos[fq + 3 : fq + 7] = [math.cos(YAW0 / 2), 0.0, 0.0, math.sin(YAW0 / 2)]
  d.qpos[qadr] = default
  mujoco.mj_forward(m, d)
  last = np.zeros(16, dtype=np.float32)
  hist = []
  top_t, fell, frames = None, False, []
  record = OUT and k == len(HS) - 1
  if record and renderer is None:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base
    cam.distance, cam.elevation, cam.azimuth = 2.4, -30.0, 90.0
  for step in range(int(T / (m.opt.timestep * DECIMATION))):
    obs, R = observe(d, last, hist)
    with torch.no_grad():
      a = policy(torch.from_numpy(obs)[None])[0].numpy()
    q_target = default[:12] + LEG_SCALE * a[:12]
    if not DC:
      d.ctrl[act[:12]] = q_target
    d.ctrl[act[12:]] = WHEEL_SCALE * a[12:]
    last = a.astype(np.float32)
    for _ in range(DECIMATION):
      if DC:
        d.ctrl[act[:12]] = dc_torque(d, q_target)
      mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    if viewer is not None:
      if not viewer.is_running():
        sys.exit(0)
      t0 = time.time()
      with viewer.lock():
        vd.qpos[:] = d.qpos
        vd.qvel[:] = d.qvel
        mujoco.mj_forward(m, vd)
      viewer.sync()
      time.sleep(max(0.0, m.opt.timestep * DECIMATION - (time.time() - t0)))
    p = d.xpos[base]
    dist = max(abs(p[0] - o[0]), abs(p[1] - o[1]))
    tilt = math.acos(max(-1.0, min(1.0, d.xmat[base].reshape(3, 3)[2, 2])))
    if tilt > math.radians(80) and (top_t is None or BAR):
      fell = True
      break
    if BAR:
      if top_t is None and p[0] - o[0] > 2.0 + WIDTH + 0.5 and p[2] - o[2] < 0.55:
        top_t = (step + 1) * m.opt.timestep * DECIMATION
    elif top_t is None and dist > 3.15 and p[2] - o[2] > 2 * h + 0.15:
      top_t = (step + 1) * m.opt.timestep * DECIMATION
    if record:
      renderer.update_scene(d, cam)
      frames.append(renderer.render())
  what = "crossed" if BAR else "top"
  res = (
    f"{what} at {top_t:.1f} s"
    if top_t is not None
    else ("FELL" if fell else f"not {what}")
  )
  if BAR and fell and top_t is not None:
    res += ", then FELL"
  print(
    f"  h {h:.3f} m (row {row}): {res}; final base z rel. floor {d.xpos[base][2] - o[2]:.2f}"
  )
  if record:
    import mediapy

    mediapy.write_video(OUT, frames, fps=int(round(1 / (m.opt.timestep * DECIMATION))))
    print(f"  video -> {OUT}")

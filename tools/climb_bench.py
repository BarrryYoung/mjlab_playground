"""Climbing stress bench for the Go2W Climb policies.

climb_eval.py asks one question (pit centre, square to the wall, vx 0.5, can it
get up?). This one builds the env once and asks it under many conditions, and
also reports *how* the robot got up.

Every env is on the climbing column, E envs per row, ROWS rows spanning HRANGE
(two rises of h per tile, 2 m of floor before the first). Deterministic policy
(mean action), no random pushes, torch seed fixed per scenario so the per-env
startup draws (wheel friction 0.5-1.0, PD gains, encoder bias) are identical
across scenarios unless a scenario overrides them.

Per scenario, one line:
  top% by height band, fall% (tilt > 80 deg before the top), median time to
  the top, and over the successful climbs: share that touched the terrain with
  the trunk / thighs / shanks, the 95th-percentile peak trunk force, and the
  share of leg-joint steps at >= 97% of the DC-motor torque curve.

Scenarios (pass names as extra args, default: all):
  base      pit centre, facing +x (yaw +-5 deg), vx 0.5, heading 0
  vx0.3 vx0.8 vx1.0          command speed
  yaw30     start yaw +-30 deg, heading command 0
  near      start 0.55 m from the wall (front wheel ~0.15 m away), no run-up
  corner    start 1.4 m off-centre: climb 0.6 m from the side wall
  fric0.4 fric0.3            wheel friction (Ascento: grip on the riser face
                             is a sim-only strategy)
  mass+3    base +3 kg (training +-1; MUJICA trains -1..+3)
  weak0.85  leg torque limits x0.85
  noise     actor obs noise on (training levels)
  push      0.5 m/s lateral push when the front reaches the first rise
  stopmid   command 0 for 3 s once the base is 0.4 h up the first rise,
            then vx 0.5 again: does it hold on the ledge / still get up?
  stoptop   command 0 on reaching the top, held 4 s (climb bit still on)
  stoptop0  same, with the climb bit switched off (the deploy procedure)
  descend   start on top facing the pit, vx 0.5, until on the pit floor
  bit0      climb bit forced off the whole time (bit tasks): is the bit needed?

usage: python tools/climb_bench.py <task> <run_glob>@<iter> [scenario ...]
env:   E (8)  ROWS (14)  HRANGE ("0.30,0.62")
"""

import contextlib
import io
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.actuator import DcMotorActuator, IdealPdActuator  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D
E = int(os.environ.get("E", "8"))
ROWS = int(os.environ.get("ROWS", "14"))
HLO, HHI = (float(x) for x in os.environ.get("HRANGE", "0.30,0.62").split(","))
CLIMB = "high_platform"
BANDS = ((0.30, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.62))
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
ALL = (
  "base vx0.3 vx0.8 vx1.0 yaw30 near corner fric0.4 fric0.3 mass+3 weak0.85 noise "
  "push stopmid stoptop stoptop0 descend bit0"
).split()

task, spec = sys.argv[1], sys.argv[2]
scenarios = sys.argv[3:] or ALL

cfg = load_env_cfg(task, play=True)
gen = cfg.scene.terrain.terrain_generator
gen.num_rows = ROWS
gen.sub_terrains[CLIMB].step_height_range = (HLO, HHI)
cfg.scene.num_envs = ROWS * E
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.observations["actor"].enable_corruption = True  # toggled per scenario below
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
N = env.num_envs
terrain = env.scene.terrain
robot = env.scene["robot"]
names = list(gen.sub_terrains)
col = names.index(CLIMB)
flat_col = names.index("flat")
row_of = torch.arange(N, device=D) // E
ids_all = torch.arange(N, device=D)
has_bit = "climb_mode" in env.observation_manager.active_terms["actor"]

# Obs noise: built in, removed here unless a scenario wants it.
actor_terms = env.observation_manager._group_obs_term_cfgs["actor"]
noise_cfgs = [t.noise for t in actor_terms]


def set_noise(on: bool) -> None:
  for t, n in zip(actor_terms, noise_cfgs, strict=True):
    t.noise = n if on else None


# Commands: fixed per env, edited in place by the scenarios.
term = env.command_manager.get_term("twist")
cmd_vx = torch.zeros(N, device=D)
cmd_heading = torch.zeros(N, device=D)


def _fixed(self, ids):
  self.vel_command_b[ids, 0] = cmd_vx[ids]
  self.vel_command_b[ids, 1:] = 0.0
  self.heading_target[ids] = cmd_heading[ids]
  self.is_heading_env[ids] = True
  self.is_standing_env[ids] = False
  self.is_world_env[ids] = False
  self.is_forward_env[ids] = False
  if hasattr(self, "_climb"):  # ClimbAwareBucketVelocityCommand: heading control on
    self._climb[ids] = True
    self._axis_pending[ids] = False


term._resample_command = _fixed.__get__(term)


def set_cmd(ids, vx):
  cmd_vx[ids] = vx
  term.vel_command_b[ids, 0] = vx


# Model fields touched by scenarios, restored after each.
wheel_local = torch.tensor(robot.find_geoms(".*_wheel_collision")[0], device=D)
wheel_gid = robot.indexing.geom_ids.to(D)[wheel_local].long()
base_bid = int(robot.indexing.body_ids[robot.find_bodies("base_link")[0][0]])
fric0 = env.sim.model.geom_friction[:, wheel_gid, 0].clone()
mass0 = env.sim.model.body_mass[:, base_bid].clone()
leg_acts = [a for a in robot.actuators if isinstance(a, IdealPdActuator)]
limits0 = [
  (
    a.force_limit.clone(),
    a.saturation_effort.clone() if isinstance(a, DcMotorActuator) else None,
  )
  for a in leg_acts
]
# Builtin (MuJoCo <position>) leg actuators keep their limit in actuator_forcerange.
leg_ctrl = torch.tensor(
  [
    int(c)
    for a in robot.actuators
    if not isinstance(a, IdealPdActuator) and "wheel" not in a.target_names[0]
    for c in a.global_ctrl_ids
  ],
  device=D,
  dtype=torch.long,
)
frange0 = (
  env.sim.model.actuator_forcerange[:, leg_ctrl].clone() if len(leg_ctrl) else None
)


def set_strength(s: float) -> None:
  for a, (fl, sat) in zip(leg_acts, limits0, strict=True):
    a.force_limit[:] = fl * s
    if sat is not None:
      a.saturation_effort[:] = sat * s
      a._vel_at_effort_lim = a.velocity_limit_motor * (
        1 + a.force_limit / a.saturation_effort
      )
  if frange0 is not None:
    env.sim.model.actuator_forcerange[:, leg_ctrl] = frange0 * s


def restore() -> None:
  env.sim.model.geom_friction[:, wheel_gid, 0] = fric0
  env.sim.model.body_mass[:, base_bid] = mass0
  set_strength(1.0)
  set_noise(False)


# Contact sensors and the DC curve used for the torque-saturation share.
touch = {k: env.scene[f"{k}_ground_touch"] for k in ("trunk", "thigh", "shank")}
jn = robot.joint_names
leg_j = [i for i, n in enumerate(jn) if "wheel" not in n]
leg_jt = torch.tensor(leg_j, device=D)
sat_tau = torch.tensor([35.55 if "calf" in jn[i] else 23.5 for i in leg_j], device=D)
sat_vel = torch.tensor([20.0 if "calf" in jn[i] else 30.0 for i in leg_j], device=D)


def touched(k: str) -> torch.Tensor:
  f = touch[k].data.force_history  # [B, N, H, 3]
  return (f.norm(dim=-1) > 1.0).flatten(1).any(dim=1)


def trunk_force() -> torch.Tensor:
  return touch["trunk"].data.force_history.norm(dim=-1).flatten(1).max(dim=1).values


def place(dx, dy, yaw0, yaw_jit, z_abs=None):
  env.reset()
  terrain.terrain_types[:] = col
  terrain.terrain_levels[:] = row_of
  terrain.env_origins[:] = terrain.terrain_origins[row_of, col]
  st = robot.data.default_root_state.clone()
  yaw = yaw0 + (torch.rand(N, device=D) * 2 - 1) * yaw_jit
  st[:, 0] = terrain.env_origins[:, 0] + dx
  st[:, 1] = terrain.env_origins[:, 1] + dy
  st[:, 2] = terrain.env_origins[:, 2] + 0.42 if z_abs is None else z_abs
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
  return obs


def run(policy, name: str):
  torch.manual_seed(0)
  restore()
  vx, T = 0.5, 15.0
  dx = dy = 0.0
  yaw0, yaw_jit, z_abs = 0.0, math.radians(5), None
  if name.startswith("vx"):
    vx = float(name[2:])
    T = 15.0 if vx >= 0.5 else 20.0
  elif name == "yaw30":
    yaw_jit = math.radians(30)
  elif name == "near":
    dx = 1.45
  elif name == "corner":
    dy = 1.4 * torch.where(ids_all % 2 == 0, 1.0, -1.0)
  elif name.startswith("fric"):
    env.sim.model.geom_friction[:, wheel_gid, 0] = float(name[4:])
  elif name == "mass+3":
    env.sim.model.body_mass[:, base_bid] = mass0 + 3.0
  elif name.startswith("weak"):
    set_strength(float(name[4:]))
  elif name == "noise":
    set_noise(True)
  elif name == "stopmid":
    T = 20.0
  elif name in ("stoptop", "stoptop0"):
    T = 19.0
  elif name == "descend":
    dx, yaw0, z_abs, T = 3.5, math.pi, 0.42, 12.0
  elif name == "bit0" and not has_bit:
    return None
  cmd_heading[:] = yaw0
  cmd_vx[:] = vx

  obs = place(dx, dy, yaw0, yaw_jit, z_abs)
  if name == "bit0":
    terrain.terrain_types[:] = flat_col
  o = terrain.env_origins.clone()
  h = -o[:, 2] / 2
  steps = int(T / env.step_dt)
  top = torch.zeros(N, dtype=torch.bool, device=D)
  fell = torch.zeros_like(top)
  t_top = torch.full((N,), float("nan"), device=D)
  tch = {k: torch.zeros_like(top) for k in touch}
  f_trunk = torch.zeros(N, device=D)
  sat_n = torch.zeros(N, device=D)
  n_climb = torch.zeros(N, device=D)
  # stop scenarios
  stop_t0 = torch.full((N,), -1, dtype=torch.long, device=D)
  hold_xy = torch.zeros(N, 2, device=D)
  drift = torch.zeros(N, device=D)
  slid = torch.zeros(N, device=D)
  z_at_stop = torch.zeros(N, device=D)
  pushed = torch.zeros_like(top)
  hold = 150 if name == "stopmid" else 200
  with torch.no_grad():
    for k in range(steps):
      obs, *_ = env.step(policy(obs["actor"]))
      p = robot.data.root_link_pos_w
      d = torch.maximum((p[:, 0] - o[:, 0]).abs(), (p[:, 1] - o[:, 1]).abs())
      z = p[:, 2] - o[:, 2]
      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      fell |= (tilt > math.radians(80)) & ~top
      if name == "descend":
        new_top = (d < 1.5) & (z < 0.6) & ~fell & ~top
      else:
        new_top = (d > 3.15) & (z > 2 * h + 0.15) & ~fell & ~top
      t_top[new_top] = (k + 1) * env.step_dt
      # how it climbs: only while still on the way up
      active = ~top & ~fell
      for key in touch:
        tch[key] |= touched(key) & active
      f_trunk = torch.where(active, torch.maximum(f_trunk, trunk_force()), f_trunk)
      v = robot.data.joint_vel[:, leg_jt]
      tau = robot.data.qfrc_actuator[:, leg_jt]
      cap = (sat_tau * (1 - v.abs() / sat_vel)).clamp(min=0.0).clamp(max=sat_tau)
      sat_n += ((tau.abs() >= 0.97 * cap).float().mean(dim=1)) * active
      n_climb += active.float()
      top |= new_top

      if name == "push":
        go = ~pushed & (d > 1.6) & ~fell
        if go.any():
          gi = ids_all[go]
          sign = torch.where(gi % 2 == 0, 1.0, -1.0)
          vel = robot.data.root_link_vel_w[gi].clone()
          vel[:, 1] += 0.5 * sign
          robot.write_root_link_velocity_to_sim(vel, env_ids=gi)
          pushed |= go
      elif name == "stopmid":
        start = (stop_t0 < 0) & (z > 0.42 + 0.4 * h) & ~top & ~fell
        if start.any():
          stop_t0[start] = k
          z_at_stop[start] = z[start]
          set_cmd(ids_all[start], 0.0)
        holding = (stop_t0 >= 0) & (k - stop_t0 < hold)
        slid = torch.where(holding, torch.maximum(slid, z_at_stop - z), slid)
        resume = (stop_t0 >= 0) & (k - stop_t0 == hold)
        if resume.any():
          set_cmd(ids_all[resume], vx)
      elif name in ("stoptop", "stoptop0"):
        start = (stop_t0 < 0) & top
        if start.any():
          stop_t0[start] = k
          hold_xy[start] = p[start, :2]
          set_cmd(ids_all[start], 0.0)
          if name == "stoptop0":
            terrain.terrain_types[ids_all[start]] = flat_col
        holding = (stop_t0 >= 0) & (k - stop_t0 <= hold)
        drift = torch.where(holding, (p[:, :2] - hold_xy).norm(dim=1), drift)
        fell |= (tilt > math.radians(80)) & (stop_t0 >= 0)

  return dict(
    h=h,
    top=top,
    fell=fell,
    t_top=t_top,
    tch=tch,
    f_trunk=f_trunk,
    sat=sat_n / n_climb.clamp(min=1),
    stop_t0=stop_t0,
    drift=drift,
    slid=slid,
    done_hold=(stop_t0 >= 0) & (steps - 1 - stop_t0 >= hold),
  )


def report(name, r):
  if r is None:
    print(f"  {name:9s} (no climb bit in this task)")
    return
  h, top = r["h"], r["top"]
  bands = []
  for lo, hi in BANDS:
    s = (h >= lo) & (h < hi)
    bands.append(f"{100 * top[s].float().mean():4.0f}" if s.any() else "   -")
  ok = top

  def pc(m):
    return f"{100 * (m & ok).float().sum() / ok.float().sum().clamp(min=1):4.0f}"

  tt = r["t_top"][ok]
  med = f"{tt.median().item():4.1f}" if len(tt) else "   -"
  f95 = r["f_trunk"][ok].quantile(0.95).item() if ok.sum() > 1 else float("nan")
  line = (
    f"  {name:9s} {' '.join(bands)}   {100 * r['fell'].float().mean():4.0f}  {med}"
    f"   {pc(r['tch']['trunk'])} {pc(r['tch']['thigh'])} {pc(r['tch']['shank'])}"
    f"  {f95:6.0f}  {100 * r['sat'][ok].mean():5.1f}"
  )
  if name in ("stoptop", "stoptop0"):
    m = r["done_hold"]
    line += (
      f"   | held {int(m.sum())}: drift p50 {100 * r['drift'][m].median():.0f} cm"
      f" p90 {100 * r['drift'][m].quantile(0.9):.0f} cm"
    )
  if name == "stopmid":
    m = r["stop_t0"] >= 0
    line += f"   | stopped {int(m.sum())}: slid back p90 {100 * r['slid'][m].quantile(0.9):.0f} cm"
  print(line, flush=True)


base = Path("logs/rsl_rl") / LOGS[task]
g, it = spec.split("@")
ck = sorted(base.glob(g))[-1] / f"model_{it}.pt"
policy = gb._load_policy(ck)
print(
  f"\n{ck.parent.name} @ {it}   {task}   E {E}/row  rows {ROWS}  h {HLO}-{HHI} m  bit {has_bit}"
)
print(
  "  top% by h:  0.30- 0.45- 0.50- 0.55-  fall% t_top  | over climbs: touch% trunk thigh shank"
  "  F_trunk p95 N  at-torque-limit %"
)
for name in scenarios:
  report(name, run(policy, name))

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import feet_swing_height
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _scaled_std(cmd_mag: torch.Tensor, std: float, cmd_scaled: bool, std_min: float):
  """std, or clamp(|cmd|, std_min, std) for nonzero commands when cmd_scaled.

  With a fixed std a small command is cheap to ignore: wz 0.1 at std 0.71 pays
  0.98 for not moving at all, and the policy stood still below wz 0.15 / vy 0.05.
  Scaled, ignoring any command below std pays exp(-1) = 0.37. Zero commands keep
  std, so standing is scored as before. Only the commanded axes use it; the
  off-axis parts (vz; roll / pitch rate) keep std, else the body rocking of a
  step would cost as much as ignoring the command.
  """
  if not cmd_scaled:
    return std
  return torch.where(cmd_mag > 0.0, cmd_mag.clamp(min=std_min, max=std), std)


def track_linear_velocity_scaled(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  cmd_scaled: bool = False,
  std_min: float = 0.05,
) -> torch.Tensor:
  """mjlab's track_linear_velocity; std shrinks with |cmd xy| when cmd_scaled."""
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  s = _scaled_std(command[:, :2].norm(dim=1), std, cmd_scaled, std_min)
  return torch.exp(-xy_error / s**2 - z_error / std**2)


def track_angular_velocity_scaled(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  cmd_scaled: bool = False,
  std_min: float = 0.05,
) -> torch.Tensor:
  """mjlab's track_angular_velocity (no heading); std shrinks with |wz cmd|."""
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  s = _scaled_std(command[:, 2].abs(), std, cmd_scaled, std_min)
  return torch.exp(-z_error / s**2 - xy_error / std**2)


def stand_still_wheel_vel_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  std: float = 1.0,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """exp(-sum(wheel_vel^2) / std^2), only where |command| < command_threshold.

  Bounded positive reward, not an L2 penalty: a penalty large enough to stop the
  wheels made the per-step return negative and the policy learned to fall over
  (fell_over has no termination penalty). Hard gate, so slow-speed tracking is
  left alone; threshold matches the pose term's walking_threshold.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  standing = torch.norm(command, dim=1) < command_threshold
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.exp(-torch.sum(torch.square(wheel_vel), dim=1) / std**2) * standing


def joint_torques_l2_straight(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  step_threshold: float = 0.05,
) -> torch.Tensor:
  """sum(actuator_force^2), only where |vy| + |wz| < step_threshold.

  That is vx-only or standing, where the legs are not asked to step. While
  stepping (lateral / turning) the torques are left alone.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  straight = (command[:, 1].abs() + command[:, 2].abs()) < step_threshold
  tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  return torch.sum(torch.square(tau), dim=1) * straight


class leg_electrical_power:
  """Electrical power of the leg motors in W: sum(max(tau * qd, 0) + c_j * tau^2).

  - max(tau * qd, 0): mechanical power, regeneration not credited.
  - c_j * tau^2: copper loss (I^2 R with tau = K I, so c = R / K^2 at the joint).
    c_j is fitted on the real Go2-W (2026-11-10 log 141528, 1 s bins, battery
    power_v * power_a against tau_est, R^2 0.997): hip 0.53, thigh 0.60,
    calf 0.17 W/(N·m)^2 (the calf has an extra reduction stage), plus ~22 W
    idle that is left out here. Copper loss dominates: holding a hip at 14 N·m
    costs ~100 W, walking does 10-40 W of mechanical work in total.

  coeffs maps a joint-name regex to c_j; actuators not matched are ignored.
  Actuator names must equal their joint names (true for Go2-W).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset = env.scene[cfg.params.get("asset_name", "robot")]
    coeffs: dict[str, float] = cfg.params["coeffs"]
    act_ids, names, c = [], [], []
    for pattern, value in coeffs.items():
      ids, found = asset.find_actuators(pattern)
      assert found, f"leg_electrical_power: no actuator matches {pattern!r}"
      act_ids += list(ids)
      names += list(found)
      c += [value] * len(ids)
    joint_ids, joint_names = asset.find_joints(names, preserve_order=True)
    assert list(joint_names) == names, (names, joint_names)
    dev = env.device
    self.act_ids = torch.tensor(act_ids, device=dev, dtype=torch.long)
    self.joint_ids = torch.tensor(joint_ids, device=dev, dtype=torch.long)
    self.c = torch.tensor(c, device=dev, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    coeffs: dict[str, float],
    asset_name: str = "robot",
  ) -> torch.Tensor:
    del coeffs  # Resolved in __init__.
    asset = env.scene[asset_name]
    tau = asset.data.actuator_force[:, self.act_ids]
    qd = asset.data.joint_vel[:, self.joint_ids]
    mech = torch.clamp(tau * qd, min=0.0).sum(1)
    heat = (self.c * torch.square(tau)).sum(1)
    log = env.extras["log"]
    log["Metrics/leg_power_mech_w"] = mech.mean()
    log["Metrics/leg_power_heat_w"] = heat.mean()
    return mech + heat


def stand_still_wheel_vel_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """sum(wheel_vel^2), only where |command| < command_threshold.

  Lower-variance alternative to stand_still_wheel_vel_exp: under exploration
  noise the exp kernel jumps between 0 and w every step, which the critic cannot
  predict and the shared advantage spreads onto the legs (leg std doubled).
  Keep the weight small (-0.01): at -1.0 the per-step return went negative and
  the policy learned to fall over.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  standing = torch.norm(command, dim=1) < command_threshold
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(wheel_vel), dim=1) * standing


class stand_still_wheel_vel_avg:
  """sum(ema(wheel_vel)^2), only where |command| < command_threshold.

  stand_still_wheel_vel_l2 penalizes the *sampled* wheel speed, ~97% of which is
  exploration noise (wheel std ~0.6 x scale 5): the deterministic policy was
  already at 0.05 rad/s while the curve sat on the noise floor. From scratch
  (wheel std 1.0) that floor made standing a net loss and stalled the legs.
  The noise is redrawn every step and averages out over time; a constant bias
  (the wheels creeping one way) does not. tau ~0.5 s cuts the floor ~N-fold
  and keeps the bias. The average restarts at 0 on entering the gate, so the
  first ~tau after a stop command is a ramp, not a full penalty on braking.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]  # Resolved before the term is built.
    num_wheels = (
      env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].shape[1]
    )
    self.avg = torch.zeros((env.num_envs, num_wheels), device=env.device)
    self.prev_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.alpha = env.step_dt / cfg.params.get("tau", 0.5)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.avg[env_ids] = 0.0
    self.prev_standing[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    tau: float = 0.5,
    kernel: str = "l2",
    max_vel: float | None = None,
  ) -> torch.Tensor:
    del tau  # Read once in __init__.
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    standing = torch.norm(command, dim=1) < command_threshold
    self.avg[standing & ~self.prev_standing] = 0.0
    self.prev_standing = standing
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    self.avg += self.alpha * (wheel_vel - self.avg)
    num_standing = torch.clamp(standing.sum(), min=1)
    env.extras["log"]["Metrics/stand_wheel_vel_avg"] = (
      torch.sum(self.avg.abs().mean(dim=1) * standing) / num_standing
    )
    # l1: the pull of l2 (2w) fades near zero, so sa002/sa003 drifted 0.08-0.28
    # rad/s late in training; |w| keeps a constant pull all the way down.
    # max_vel caps the penalty: on rough, a flipped robot spinning its wheels in
    # the air gave -76 per episode in one batch and blew up the value loss.
    avg = self.avg if max_vel is None else self.avg.clamp(-max_vel, max_vel)
    per_wheel = avg.abs() if kernel == "l1" else torch.square(avg)
    return torch.sum(per_wheel, dim=1) * standing


class stand_still_wheel_vel_avg_cmd_ema:
  """sum(ema(wheel_vel)^2), gated on the command *and* its own ema.

  stand_still_wheel_vel_avg opens the gate the step the command drops to zero
  and only softens braking by restarting its average. Here the gate also waits
  for ema(|command|) to fall below the threshold, so the grace period scales
  with the speed being dropped: cmd_tau = 0.2 / ln(1.0 / 0.05) ~ 0.067 s gives
  0.2 s from 1.0, 0.25 s from 2.0, 0.05 s from 0.1. The current command must be
  below the threshold too, or starting to move would be penalized while the
  ema is still low. The wheel average restarts at 0 when the gate opens, so the
  grace period is followed by a ~tau ramp: a continuous average would still
  carry ~0.5 s of braking tail at that point.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]  # Resolved before the term is built.
    num_wheels = (
      env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].shape[1]
    )
    self.avg = torch.zeros((env.num_envs, num_wheels), device=env.device)
    self.cmd_avg = torch.zeros(env.num_envs, device=env.device)
    self.prev_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.alpha = env.step_dt / cfg.params.get("tau", 0.5)
    # Exact form: cmd_tau is short enough that dt / tau would be ~0.3, off by 15%.
    self.cmd_alpha = 1.0 - math.exp(-env.step_dt / cfg.params.get("cmd_tau", 0.067))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.avg[env_ids] = 0.0
    self.cmd_avg[env_ids] = 0.0
    self.prev_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    tau: float = 0.5,
    cmd_tau: float = 0.067,
  ) -> torch.Tensor:
    del tau, cmd_tau  # Read once in __init__.
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    cmd_norm = torch.norm(command, dim=1)
    self.cmd_avg += self.cmd_alpha * (cmd_norm - self.cmd_avg)
    active = (cmd_norm < command_threshold) & (self.cmd_avg < command_threshold)
    self.avg[active & ~self.prev_active] = 0.0
    self.prev_active = active
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    self.avg += self.alpha * (wheel_vel - self.avg)
    num_active = torch.clamp(active.sum(), min=1)
    env.extras["log"]["Metrics/stand_wheel_vel_avg_cmd_ema"] = (
      torch.sum(self.avg.abs().mean(dim=1) * active) / num_active
    )
    return torch.sum(torch.square(self.avg), dim=1) * active


def wheels_grounded(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Fraction of wheels in contact, only where |vy| + |wz| command <= threshold.

  Covers straight forward/backward driving and standing; the complement of
  feet_swing_clearance's gate, so the two never ask for opposite things.
  Contact rather than height on purpose: the unloaded wheel (RR) hovers only
  ~0.04 mm above ground when standing, invisible to a height check.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  active = (command[:, 1].abs() + command[:, 2].abs()) <= command_threshold
  in_contact = (env.scene[sensor_name].data.found > 0).float()
  return in_contact.mean(dim=1) * active


def wheel_air_time_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_air_time: float = 1.0,
) -> torch.Tensor:
  """sum over wheels of current_air_time^2 [s^2], air time capped at max_air_time.

  soft_landing and foot_swing_clearance only fire at touchdown, so a wheel that
  never comes down costs nothing: with soft_landing on (m22 @7500 -> sl1e3) the
  trot slowed from a 0.3 s to a 0.4-0.54 s period and wheels hung up to 1.8 s.
  No free allowance (a threshold would just get every swing parked at it):
  one swing of length T costs ~T^3 / (3 dt), so it trades off against
  soft_landing's per-touchdown cost and the period settles where they balance.
  Ungated. The cap only bounds the value; past it the penalty stays flat.
  """
  air_time = env.scene[sensor_name].data.current_air_time
  assert air_time is not None, f"'{sensor_name}' needs track_air_time=True."
  return torch.square(torch.clamp(air_time, max=max_air_time)).sum(dim=1)


def soft_landing_peak(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 150.0,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """sum over wheels that touched down this step of relu(peak |F| - threshold) [N].

  Replaces mjlab's soft_landing, which reads |F| once at the end of the policy
  step: the impact decays within ~15 ms, so the reading depended on which of
  the 4 substeps the wheel landed in (p50 251 N at 5 ms vs 130 N at 20 ms), and
  compute_first_contact's 1e-8 tolerance loses the 20 ms case to float32
  accumulation (contact_time 0.0200005), so ~1/4 of touchdowns were never
  penalized. Here the peak is taken over the step's substeps (sensor needs
  history_length = decimation) and the window gets half a substep of slack.
  The threshold sits above normal two-wheel support (stance peak p99 ~137 N on
  m22_sl1e3_air2 @8999), so only the impact is penalized, not body weight.
  """
  sensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force_history is not None, (
    f"'{sensor_name}' needs history_length = decimation."
  )
  assert data.current_contact_time is not None, (
    f"'{sensor_name}' needs track_air_time=True."
  )
  contact_time = data.current_contact_time
  first_contact = (contact_time > 0.0) & (
    contact_time < env.step_dt + 0.5 * env.physics_dt
  )
  peak = torch.norm(data.force_history, dim=-1).max(dim=-1).values  # [B, N]
  excess = torch.relu(peak - force_threshold) * first_contact.float()
  num_landings = torch.clamp(first_contact.float().sum(), min=1.0)
  env.extras["log"]["Metrics/landing_peak_mean"] = (
    peak * first_contact.float()
  ).sum() / num_landings
  env.extras["log"]["Metrics/landing_excess_mean"] = excess.sum() / num_landings
  cost = excess.sum(dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    cost = cost * (total > command_threshold).float()
  return cost


def wheel_lateral_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """sum over grounded wheels of (wheel-frame vy)^2, i.e. sliding along the axle.

  Rolling moves the contact point perpendicular to the axle, so it is not
  penalized; feet_slip penalizes all xy velocity and has to stay off for wheels.
  The axle is the wheel link's own y axis (wheel joint axis="0 1 0"), so it
  tilts with hip abduction and base roll. asset_cfg.body_names must be the
  wheel links in the contact sensor's geom order.
  """
  asset = env.scene[asset_cfg.name]
  contact_sensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, N, 4]
  y_axis = torch.zeros(quat.shape[:-1] + (3,), device=quat.device)
  y_axis[..., 1] = 1.0
  axle_w = quat_apply(quat, y_axis)  # [B, N, 3]
  vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]  # [B, N, 3]
  slip = torch.sum(vel_w * axle_w, dim=-1)  # [B, N]
  num_in_contact = torch.sum(in_contact)
  env.extras["log"]["Metrics/wheel_lateral_slip_mean"] = torch.sum(
    slip.abs() * in_contact
  ) / torch.clamp(num_in_contact, min=1)
  return torch.sum(torch.square(slip) * in_contact, dim=1)


class feet_swing_height_aligned(feet_swing_height):
  """Upstream feet_swing_height with the two sensors realigned, nothing else.

  The contact sensor orders its slots by MJCF order (FL, FR, RL, RR) while the
  height sensor follows its frame tuple (FR, FL, RR, RL), and upstream pairs them
  elementwise, so each landing was scored on the opposite wheel's peak. Go1 is
  unaffected because its MJCF order matches the tuple. The permutation and the
  wheel radii (collision geom size[0]) are read from the model.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    model = env.sim.mj_model
    contact_sensor = env.scene[cfg.params["sensor_name"]]
    height_sensor = env.scene[cfg.params["height_sensor_name"]]

    entity = contact_sensor.cfg.primary.entity
    geom_ids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{entity}/{s.primary_name}")
      for s in contact_sensor._slots
      if s.field_name == "found"
    ]
    frames = height_sensor.cfg.frame
    frames = frames if isinstance(frames, tuple) else (frames,)
    site_ids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f.entity}/{f.name}")
      for f in frames
    ]
    assert -1 not in geom_ids and -1 not in site_ids
    site_body = [int(model.site_bodyid[i]) for i in site_ids]
    # perm[j] = height-sensor frame on the same body as contact slot j.
    perm = [site_body.index(int(model.geom_bodyid[g])) for g in geom_ids]
    self.perm = torch.tensor(perm, device=env.device)
    self.radius = torch.tensor(
      [float(model.geom_size[g, 0]) for g in geom_ids], device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = env.scene[height_sensor_name].data.heights[:, self.perm]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      self.peak_heights * first_contact.float()
    ) / torch.clamp(num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost


class feet_swing_clearance(feet_swing_height_aligned):
  """feet_swing_height_aligned on ground clearance (frame height minus radius).

  Upstream divides the frame height by the target. The height sensor reports the
  wheel *center*, so a wheel that never lifts already reads 0.086 m and costs at
  most (0.086/0.126 - 1)^2 = 0.10 per landing; on clearance it costs 1.0.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_clearance: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    clearance = env.scene[height_sensor_name].data.heights[:, self.perm] - self.radius
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, clearance), self.peak_heights
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    # Only lateral/yaw commands need steps; vx is left to the wheels. Gating on
    # the full command made the policy trot at 4 cm under every command, vx 0.1
    # included.
    total_command = torch.abs(command[:, 1]) + torch.abs(command[:, 2])
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_clearance - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    env.extras["log"]["Metrics/peak_clearance_mean"] = torch.sum(
      self.peak_heights * first_contact.float()
    ) / torch.clamp(num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost


class diagonal_gait:
  """Reward for an even, diagonal gait, in [0, 1]; one of two cases per env.

  - straight (|vy| + |wz| <= straight_threshold: vx only or standing):
    load balance = 1 - sum |share - 1/4| / 1.5  (1 = even, 0 = one wheel),
    share = |F| / total |F| at this step. The policy rode the FR/RL diagonal
    with FL/RR carrying 1-8%. Tried and dropped: smoothing the share over
    0.5 s (a 0.25 s lift every ~3 s barely moved the score, the trot leaked
    into standing), a 0.25 s smoothed grace after a stop, and a multiplicative
    leg-still gate exp(-(leg joint-vel RMS / 0.5)^2) (action noise held it at
    ~0.27 in training and the FR/RL diagonal came back, balance 0.80 -> 0.47).
  - stepping (|vy| + |wz| > step_threshold, same gate as foot_swing_clearance):
    [one diagonal in contact, the other in the air] x mean over wheels of
    q = exp(-((T - target_air) / air_sigma)^2), T the wheel's last swing, or
    its current one once that is past target_air (so a hung wheel decays).
    The time share of the diagonal state does not depend on step frequency;
    q sets the step length.
    The first version scored smoothed load-based contact here: wheels only
    touching lightly passed for half in the air (wz 0.15 scrubbed), and a
    0.5 s window was filled best by short steps, so the gait locked at
    4.1-4.5 Hz (air ~0.11 s) for every command. Normal steps of the good runs
    (lyaw_yaw2, fsc035 sa003) are 0.12-0.16 s in the air. A hard gate at
    0.15 s paid lyaw_yaw2 ~0 (both wheels of a diagonal rarely pass it
    together) and paid most the run hanging FL/RR up, hence the soft gate.
  - in between: 0.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor = env.scene[cfg.params["sensor_name"]]
    names = [
      s.primary_name.split("_")[0] for s in sensor._slots if s.field_name == "found"
    ]
    self.diag_a = [names.index("FL"), names.index("RR")]
    self.diag_b = [names.index("FR"), names.index("RL")]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    target_air: float = 0.15,
    air_sigma: float = 0.05,
    straight_threshold: float = 1e-3,
    step_threshold: float = 0.1,
  ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    data = env.scene[sensor_name].data
    assert data.current_air_time is not None, (
      f"'{sensor_name}' needs track_air_time=True."
    )
    force = data.force.norm(dim=-1)
    total = force.sum(dim=1, keepdim=True)
    n = force.shape[1]
    share = torch.where(total > 1.0, force / total.clamp(min=1.0), 1.0 / n)

    steer = command[:, 1].abs() + command[:, 2].abs()
    straight = steer <= straight_threshold
    stepping = steer > step_threshold

    balance = 1.0 - (share - 1.0 / n).abs().sum(dim=1) / (2.0 * (1.0 - 1.0 / n))

    air = data.current_air_time
    in_air = air > 0.0
    contact = data.current_contact_time > 0.0
    swing_t = torch.where(in_air & (air > target_air), air, data.last_air_time)
    q = torch.exp(-torch.square((swing_t - target_air) / air_sigma)).mean(1)
    a, b = self.diag_a, self.diag_b
    a_up = in_air[:, a].all(1) & contact[:, b].all(1)
    b_up = in_air[:, b].all(1) & contact[:, a].all(1)
    diag_state = (a_up | b_up).float()
    diag_swing = diag_state * q

    log = env.extras["log"]
    for name, value, mask in (
      ("diag_gait_balance", balance, straight),
      ("diag_gait_state", diag_state, stepping),
      ("diag_gait_air_q", q, stepping),
    ):
      log[f"Metrics/{name}"] = (value * mask).sum() / mask.sum().clamp(min=1)
    return balance * straight + diag_swing * stepping


class diagonal_gait_v2:
  """Diagonal trot of a given period, scored on the last period, in [0, 1].

  - straight (|vy| + |wz| <= straight_threshold: vx only or standing):
    load balance = (1 - sum |share - 1/4| / 1.5) ** balance_power, share = |F| /
    total |F| at this step, as in diagonal_gait.
  - otherwise (stepping; the command deadzone keeps |vy| + |wz| at 0 or >= 0.05):
    each wheel is firmly down when c = |F| / (full_load_frac * total / 4) >=
    1 (c = 8 * share by default; a wheel resting lightly at 4% gives c ~0.3).
    A diagonal is down when c1 * c2 >= contact_threshold. Over a ring of the
    last W = period / step_dt steps:
      L_A, L_B = longest run of "diagonal down", circular (a run split by the
                 window edge counts as one), so the score does not depend on
                 where in the cycle the window starts;
      U = |run_A u run_B|, the two longest runs merged, overlap counted once.
    score = qA * qB * qU, q = exp(-((x - target) / sigma)^2),
    targets L = period / 2 and U = period. Short steps make L short, uneven
    diagonals make L_A != L_B, scrubbing on four wheels makes L = period, both
    diagonals landing together (pronk) makes U = period / 2.

  The score depends on the last period, which the critic cannot see; state()
  is fed to it (see dgv2_state). The window length is read once in __init__.
  The window only advances while the term is computed, i.e. at weight != 0.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor = env.scene[cfg.params["sensor_name"]]
    names = [
      s.primary_name.split("_")[0] for s in sensor._slots if s.field_name == "found"
    ]
    self.diag_a = [names.index("FL"), names.index("RR")]
    self.diag_b = [names.index("FR"), names.index("RL")]
    self.step_dt = env.step_dt
    self.w = max(2, round(cfg.params.get("period", 0.5) / env.step_dt))
    n, dev = env.num_envs, env.device
    self.down = torch.zeros((n, 2, self.w), dtype=torch.bool, device=dev)
    self.ptr = 0
    self.cur = torch.zeros((n, 2), device=dev)  # current run, in steps
    self.run_len = torch.zeros((n, 2), device=dev)  # L_A, L_B, in steps
    self.union = torch.zeros(n, device=dev)  # U, in steps
    self.ring = torch.arange(self.w, device=dev)
    self.idx2 = torch.arange(2 * self.w, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.down[env_ids] = False
    self.cur[env_ids] = 0.0
    self.run_len[env_ids] = 0.0
    self.union[env_ids] = 0.0

  def state(self) -> torch.Tensor:
    """[L_A, L_B, U, current run A, current run B] / W, for the critic."""
    return (
      torch.cat([self.run_len, self.union[:, None], self.cur.clamp(max=self.w)], dim=1)
      / self.w
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    period: float = 0.5,
    sigma: float = 0.05,
    contact_threshold: float = 0.8,
    full_load_frac: float = 0.5,
    straight_threshold: float = 1e-3,
    steer_deadband: float = 0.0,
    balance_power: float = 1.0,
    require_all_down: bool = False,
  ) -> torch.Tensor:
    del period  # Read once in __init__.
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    force = env.scene[sensor_name].data.force.norm(dim=-1)
    total = force.sum(dim=1, keepdim=True)
    n = force.shape[1]
    share = torch.where(total > 1.0, force / total.clamp(min=1.0), 1.0 / n)
    balance = 1.0 - (share - 1.0 / n).abs().sum(dim=1) / (2.0 * (1.0 - 1.0 / n))
    # balance_power > 1 widens the gap to stepping / diagonal support (~0.38).
    balance = balance.clamp(min=0.0) ** balance_power

    c = torch.clamp(force / (full_load_frac * total.clamp(min=1.0) / n), max=1.0)
    # require_all_down: x min(c), so any wheel in the air zeroes the straight
    # score and a light touch (4% of the load, c ~0.3) cuts it to ~0.3.
    if require_all_down:
      balance = balance * c.min(dim=1).values
    d = (
      torch.stack([c[:, self.diag_a].prod(1), c[:, self.diag_b].prod(1)], dim=1)
      >= contact_threshold
    )  # (N, 2)
    self.down[:, :, self.ptr] = d
    self.ptr = (self.ptr + 1) % self.w
    self.cur = (self.cur + 1.0) * d

    # Longest circular run: longest run in the ring laid out twice, capped at W.
    w = self.w
    xx = torch.cat([self.down, self.down], dim=-1)  # (N, 2, 2W)
    last_up = torch.where(~xx, self.idx2, -1).cummax(dim=-1).values
    run = self.idx2 - last_up  # length of the run of "down" ending at each slot
    best, end = run.max(dim=-1)
    best = best.clamp(max=w)
    # Slots of that run on the ring: p with (end - p) mod W < best.
    in_run = (end[..., None] - self.ring) % w < best[..., None]  # (N, 2, W)
    self.run_len = best.float()
    self.union = (in_run[:, 0] | in_run[:, 1]).sum(-1).float()

    dt = self.step_dt
    half = w * dt / 2.0
    q = torch.exp(-torch.square((self.run_len * dt - half) / sigma))  # (N, 2)
    q_u = torch.exp(-torch.square((self.union * dt - 2.0 * half) / sigma))
    gait = q[:, 0] * q[:, 1] * q_u

    steer = command[:, 1].abs() + command[:, 2].abs()
    straight = steer <= straight_threshold
    # deadband: the flat tasks' command deadzone keeps steer at 0 or >= 0.05,
    # but a heading controller writes wz every step and settles at a degree or
    # two, i.e. 0.001-0.05. Charge neither trot nor balance there (as a joystick
    # deadzone would): the robot is driving straight with a small correction.
    stepping = steer > max(straight_threshold, steer_deadband)

    log = env.extras["log"]
    for name, value, mask in (
      ("dgv2_balance", balance, straight),
      ("dgv2_gait", gait, stepping),
      ("dgv2_q_a", q[:, 0], stepping),
      ("dgv2_q_b", q[:, 1], stepping),
      ("dgv2_q_union", q_u, stepping),
      ("dgv2_run_a_s", self.run_len[:, 0] * dt, stepping),
      ("dgv2_run_b_s", self.run_len[:, 1] * dt, stepping),
      ("dgv2_union_s", self.union * dt, stepping),
    ):
      log[f"Metrics/{name}"] = (value * mask).sum() / mask.sum().clamp(min=1)
    return balance * straight + gait * stepping

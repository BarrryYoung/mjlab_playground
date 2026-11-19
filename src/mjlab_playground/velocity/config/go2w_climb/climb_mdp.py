"""MDP terms for the Go2W climbing task.

Everything here keys off one fact of mjlab's curriculum terrain: each
sub-terrain type owns exactly one column, so an env is on the climbing tiles
iff ``terrain.terrain_types == <column of the climbing sub-terrain>``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, GridPatternCfg
from mjlab.tasks.velocity.mdp.rewards import upright, variable_posture
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from mjlab_playground.velocity.mdp.commands import (
  BucketVelocityCommand,
  BucketVelocityCommandCfg,
)
from mjlab_playground.velocity.mdp.rewards import (
  diagonal_gait_v2,
  feet_swing_clearance,
  track_angular_velocity_scaled,
  track_linear_velocity_scaled,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
  """Grid pattern whose ray origins sit ``z_offset`` above the frame.

  A RayCastSensor casts from the frame position; ``ray_alignment`` only turns
  the rays. From the base (~0.4 m up) a 0.5 m ledge is above the ray origin, so
  rays start inside the step box and report a pit. With ``ray_alignment="yaw"``
  the offset is rotated by yaw only, i.e. stays vertical whatever the pitch.
  height_scan still reads frame z - hit z, so values are unchanged.
  """

  z_offset: float = 1.0

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.z_offset
    return offsets, directions


def climb_terrain_mask(env: ManagerBasedRlEnv, terrain_name: str) -> torch.Tensor:
  """[B] bool, True for envs whose curriculum column is ``terrain_name``.

  ``terrain_name`` may list several columns separated by "|" ("high_platform|bar").
  """
  terrain = env.scene.terrain
  gen = None if terrain is None else terrain.cfg.terrain_generator
  if gen is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  names = list(gen.sub_terrains.keys())
  cols = [names.index(n) for n in terrain_name.split("|") if n in names]
  if not cols:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if len(cols) == 1:
    return terrain.terrain_types == cols[0]
  t = terrain.terrain_types
  return torch.isin(t, torch.tensor(cols, device=t.device, dtype=t.dtype))


##
# Command
##


class ClimbAwareVelocityCommand(UniformVelocityCommand):
  """Upstream command, overridden on the climbing tiles.

  On the climbing column every resample is: vx uniform in ``climb_lin_vel_x``,
  vy = 0, and a heading target on the axis nearest the current heading (the
  pit walls are axis aligned) plus up to +-``climb_heading_noise``. So the robot
  drives straight at a wall, roughly square to it, as an operator would.
  """

  cfg: ClimbAwareVelocityCommandCfg

  def __init__(self, cfg: ClimbAwareVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    # The axis is picked from the heading, which is stale inside a reset (the
    # pose is written but forward() has not run yet), so it is deferred to the
    # next _update_command, which runs after the env's forward() call.
    self._axis_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._axis_noise = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    mask = climb_terrain_mask(self._env, self.cfg.climb_terrain_name)[env_ids]
    ids = env_ids[mask]
    if len(ids) == 0:
      return
    r = torch.empty(len(ids), device=self.device)
    self.vel_command_b[ids, 0] = r.uniform_(*self.cfg.climb_lin_vel_x)
    self.vel_command_b[ids, 1] = 0.0
    self.vel_command_b[ids, 2] = 0.0
    self._axis_noise[ids] = r.uniform_(
      -self.cfg.climb_heading_noise, self.cfg.climb_heading_noise
    )
    self._axis_pending[ids] = True
    self.is_heading_env[ids] = True
    self.is_standing_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False
    self.vel_command_w[ids] = self.vel_command_b[ids]

  def _update_command(self) -> None:
    if self._axis_pending.any():
      ids = self._axis_pending.nonzero(as_tuple=False).flatten()
      quarter = 0.5 * math.pi
      axis = torch.round(self.robot.data.heading_w[ids] / quarter) * quarter
      self.heading_target[ids] = wrap_to_pi(axis + self._axis_noise[ids])
      self._axis_pending[ids] = False
    super()._update_command()


@dataclass(kw_only=True)
class ClimbAwareVelocityCommandCfg(UniformVelocityCommandCfg):
  climb_terrain_name: str = "high_platform"
  climb_lin_vel_x: tuple[float, float] = (0.3, 0.8)
  climb_heading_noise: float = math.radians(10.0)

  def build(self, env) -> ClimbAwareVelocityCommand:
    return ClimbAwareVelocityCommand(self, env)


def terrain_in(env: ManagerBasedRlEnv, terrain_names: tuple[str, ...]) -> torch.Tensor:
  """[B] bool, True for envs whose column is one of ``terrain_names``.

  Without a terrain generator (a plane, e.g. in a flat benchmark) every env counts
  as in: the plane is the flat column.
  """
  terrain = env.scene.terrain
  gen = None if terrain is None else terrain.cfg.terrain_generator
  if gen is None or terrain.terrain_origins is None:
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  names = list(gen.sub_terrains.keys())
  cols = [names.index(n) for n in terrain_names if n in names]
  t = terrain.terrain_types
  return torch.isin(t, torch.tensor(cols, device=t.device, dtype=t.dtype))


class ClimbAwareBucketVelocityCommand(BucketVelocityCommand):
  """The flat tasks' bucket command (standing / vx / vy / wz / slow / combo),
  overridden on the climbing column as in ClimbAwareVelocityCommand: forward
  only, square to the nearest wall. BucketVelocityCommand runs without heading
  control, so the climbing envs' yaw rate is set here from their heading target.
  """

  cfg: ClimbAwareBucketVelocityCommandCfg

  def __init__(self, cfg: ClimbAwareBucketVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._climb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._axis_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._axis_noise = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    mask = climb_terrain_mask(self._env, self.cfg.climb_terrain_name)[env_ids]
    self._climb[env_ids] = mask
    ids = env_ids[mask]
    if len(ids) == 0:
      return
    r = torch.empty(len(ids), device=self.device)
    self.vel_command_b[ids, 0] = r.uniform_(*self.cfg.climb_lin_vel_x)
    self.vel_command_b[ids, 1:] = 0.0
    if self.cfg.climb_stop_prob > 0.0:
      # The operator letting go of the stick on a ledge or a bar: hold still.
      stop = torch.rand(len(ids), device=self.device) < self.cfg.climb_stop_prob
      self.vel_command_b[ids[stop], 0] = 0.0
    self._axis_noise[ids] = r.uniform_(
      -self.cfg.climb_heading_noise, self.cfg.climb_heading_noise
    )
    self._axis_pending[ids] = True
    self.is_standing_env[ids] = False
    self.vel_command_w[ids] = self.vel_command_b[ids]

  def _update_command(self) -> None:
    if self._axis_pending.any():
      ids = self._axis_pending.nonzero(as_tuple=False).flatten()
      quarter = 0.5 * math.pi
      axis = torch.round(self.robot.data.heading_w[ids] / quarter) * quarter
      self.heading_target[ids] = wrap_to_pi(axis + self._axis_noise[ids])
      self._axis_pending[ids] = False
    super()._update_command()
    ids = self._climb.nonzero(as_tuple=False).flatten()
    if len(ids):
      err = wrap_to_pi(self.heading_target[ids] - self.robot.data.heading_w[ids])
      lim = self.cfg.climb_max_yaw_rate
      self.vel_command_b[ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * err, min=-lim, max=lim
      )


@dataclass(kw_only=True)
class ClimbAwareBucketVelocityCommandCfg(BucketVelocityCommandCfg):
  climb_terrain_name: str = "high_platform"
  climb_lin_vel_x: tuple[float, float] = (0.3, 0.8)
  climb_heading_noise: float = math.radians(10.0)
  climb_max_yaw_rate: float = 1.0
  # Share of climbing-column resamples that are a stop (vx 0, heading held).
  climb_stop_prob: float = 0.0

  def build(self, env) -> ClimbAwareBucketVelocityCommand:
    return ClimbAwareBucketVelocityCommand(self, env)


##
# Rewards
##


def track_lin_vel_xy_yaw_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Planar velocity tracking in the yaw frame.

  mjlab's track_linear_velocity uses the body frame and adds vz^2 to the error,
  i.e. it pays the robot not to rise -- the opposite of what a climb needs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w
  )
  error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Yaw-rate tracking about world z, without mjlab's roll/pitch-rate error."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return torch.exp(-error / std**2)


def track_lin_vel_climb_or_flat(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  climb_terrain_name: str,
  climb_scale: float = 2.0,
  std_min: float = 0.1,
) -> torch.Tensor:
  """Climbing column: yaw-frame planar tracking x climb_scale (the Climb tasks'
  term at its weight 2). Elsewhere: the flat tasks' term (body frame, vz in the
  error, std shrinking with |cmd| down to std_min), at weight 1."""
  climb = climb_terrain_mask(env, climb_terrain_name)
  r_climb = track_lin_vel_xy_yaw_exp(env, std, command_name)
  r_flat = track_linear_velocity_scaled(env, std, command_name, True, std_min)
  return torch.where(climb, climb_scale * r_climb, r_flat)


def track_ang_vel_climb_or_flat(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  climb_terrain_name: str,
  std_min: float = 0.1,
) -> torch.Tensor:
  """Climbing column: world yaw rate only. Elsewhere: the flat tasks' term (body
  yaw rate, roll / pitch rate in the error, cmd-scaled std)."""
  climb = climb_terrain_mask(env, climb_terrain_name)
  r_climb = track_ang_vel_z_world_exp(env, std, command_name)
  r_flat = track_angular_velocity_scaled(env, std, command_name, True, std_min)
  return torch.where(climb, r_climb, r_flat)


class diagonal_gait_v2_on(diagonal_gait_v2):
  """The flat tasks' dgv2, paid only on the ``posture_terrains`` columns."""

  def __call__(self, env: ManagerBasedRlEnv, posture_terrains: tuple[str, ...], **kw):  # type: ignore[override]
    return super().__call__(env, **kw) * terrain_in(env, posture_terrains).float()


class feet_swing_clearance_on(feet_swing_clearance):
  """The flat tasks' swing clearance cost, only on the ``posture_terrains`` columns."""

  def __call__(self, env: ManagerBasedRlEnv, posture_terrains: tuple[str, ...], **kw):  # type: ignore[override]
    return super().__call__(env, **kw) * terrain_in(env, posture_terrains).float()


def lin_vel_z_l2_off_climb(
  env: ManagerBasedRlEnv,
  climb_terrain_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """World vz^2, everywhere except the climbing tiles."""
  asset: Entity = env.scene[asset_cfg.name]
  off = (~climb_terrain_mask(env, climb_terrain_name)).float()
  return torch.square(asset.data.root_link_lin_vel_w[:, 2]) * off


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


class upright_off_climb(upright):
  """mjlab's upright, zeroed on the climbing tiles (a climb has to pitch up)."""

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    std: float,
    climb_terrain_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    r = super().__call__(env, std, asset_cfg, terrain_sensor_names)
    return r * (~climb_terrain_mask(env, climb_terrain_name)).float()


class variable_posture_off_climb(variable_posture):
  """mjlab's variable_posture, zeroed on the climbing tiles."""

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    climb_terrain_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    r = super().__call__(
      env,
      std_standing,
      std_walking,
      std_running,
      asset_cfg,
      command_name,
      walking_threshold,
      running_threshold,
    )
    return r * (~climb_terrain_mask(env, climb_terrain_name)).float()


##
# Events
##


def scale_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale kp and kd of every explicit-PD (IdealPd / DC motor) actuator group.

  mjlab's dr.pd_gains indexes the actuator groups with ids resolved against the
  per-joint actuator names and raises on the builtin wheel actuators; this only
  touches the PD groups and leaves the wheels alone.
  """
  from mjlab.actuator import IdealPdActuator

  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  for act in asset.actuators:
    if not isinstance(act, IdealPdActuator):
      continue
    assert act.default_stiffness is not None and act.default_damping is not None
    n = act.default_stiffness.shape[1]
    kp = torch.empty(len(env_ids), n, device=env.device).uniform_(*scale_range)
    kd = torch.empty(len(env_ids), n, device=env.device).uniform_(*scale_range)
    act.set_gains(
      env_ids,
      kp=act.default_stiffness[env_ids] * kp,
      kd=act.default_damping[env_ids] * kd,
    )


##
# Observations
##


def climb_indicator(env: ManagerBasedRlEnv, terrain_name: str) -> torch.Tensor:
  """[B, 1]: 1 on the climbing tiles. The blind policy's 'climb mode' switch."""
  return climb_terrain_mask(env, terrain_name).float().unsqueeze(1)


def contact_found(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B, N] 0/1 per primary of a contact sensor (critic only)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()

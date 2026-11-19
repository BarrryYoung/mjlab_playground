"""Velocity-task metrics: logged as Episode_Metrics/<name>, never trained on.

Each returns a per-env value in physical units, shape (B,).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_err(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """||v_xy^cmd - v_xy|| in the base frame [m/s]. Same quantities as the
  track_linear_velocity reward, without its z term and exp kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  return torch.norm(command[:, :2] - actual[:, :2], dim=1)


def wheel_air_time_max(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Longest current air time over the four wheels [s]."""
  air_time = env.scene[sensor_name].data.current_air_time
  assert air_time is not None, f"'{sensor_name}' needs track_air_time=True."
  return air_time.max(dim=1).values


def track_yaw_rate_err(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|w_z^cmd - w_z| in the base frame [rad/s]. Same quantities as the
  track_angular_velocity reward, without its roll/pitch term and exp kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  return torch.abs(command[:, 2] - actual[:, 2])

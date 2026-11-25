"""Rewards for the Go2W rough ZeroStd tasks (11-24).

Same as go2w_rough.rough_mdp.track_{linear,angular}_velocity_terrain, except that
on ``full_terrains`` a zero command uses the scaled std too:
std = clamp(|cmd|, std_min, std) for every command, 0 included.

Before (mdp.rewards._scaled_std) a zero command kept the full std (0.5 / 0.71):
a 1 cm/s creep on a stand command cost 0.04 % of the term, 2 cm/s 0.16 %, so the
policy had no reason to stop rolling (sim 11-24: ~1 cm/s, 23 cm in 20 s). At
std_min 0.1 they cost 1 % / 3.9 %. It also removes the jump between a command of
0 (std 0.5) and 0.01 (std 0.1). The angular term counts wz alone, so driving
straight (wz 0) on the full terrains is also held to 0.1 rad/s now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_playground.velocity.config.go2w_climb.climb_mdp import terrain_in

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _std_zero_too(cmd_mag: torch.Tensor, std: float, std_min: float) -> torch.Tensor:
  return cmd_mag.clamp(min=std_min, max=std)


def track_linear_velocity_terrain_zerostd(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  full_terrains: tuple[str, ...] = ("flat",),
  cmd_scaled: bool = True,
  std_min: float = 0.1,
) -> torch.Tensor:
  assert cmd_scaled, "zerostd is the cmd-scaled std with zero commands included"
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  full = terrain_in(env, full_terrains)
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2]) * full.float()
  s = torch.where(full, _std_zero_too(command[:, :2].norm(dim=1), std, std_min), std)
  return torch.exp(-xy_error / s**2 - z_error / std**2)


def track_angular_velocity_terrain_zerostd(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  full_terrains: tuple[str, ...] = ("flat",),
  cmd_scaled: bool = True,
  std_min: float = 0.1,
) -> torch.Tensor:
  assert cmd_scaled, "zerostd is the cmd-scaled std with zero commands included"
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  full = terrain_in(env, full_terrains)
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1) * full.float()
  s = torch.where(full, _std_zero_too(command[:, 2].abs(), std, std_min), std)
  return torch.exp(-z_error / s**2 - xy_error / std**2)

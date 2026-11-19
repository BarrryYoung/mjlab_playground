from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_pos_rel_without_wheel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """joint_pos_rel with wheel entries zeroed (robot_lab's term of the same name).

  Wheel angle is unbounded, so the slot is kept only to match robot_lab's 16-wide
  layout; it carries no information.
  """
  asset = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  out = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
  wheel_cols = [i for i, n in enumerate(asset_cfg.joint_names) if "wheel" in n]
  out[:, wheel_cols] = 0.0
  return out


def dgv2_state(env: ManagerBasedRlEnv, reward_name: str = "dgv2") -> torch.Tensor:
  """diagonal_gait_v2's window summary (5 values), critic only.

  The observation manager is built before the reward manager and calls each term
  once for its size, so zeros are returned until the reward exists.
  """
  reward_manager = getattr(env, "reward_manager", None)
  if reward_manager is None:
    return torch.zeros((env.num_envs, 5), device=env.device)
  return reward_manager.get_term_cfg(reward_name).func.state()

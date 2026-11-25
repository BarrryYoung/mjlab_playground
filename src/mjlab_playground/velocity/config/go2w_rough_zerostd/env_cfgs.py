"""Go2W rough ZeroStd (11-24): Straight-H6-LowNoise-Ht0003 with the velocity
tracking std scaled for zero commands too (std_min 0.1), aimed at the policy
creeping ~1 cm/s on a stand command. Two tasks:

  - ZeroStd:          rewards changed only; actor 342 as before.
  - ZeroStd-WheelPos: the same plus the WheelPos actor (346, wheel_vel_pos).

New task family; config/go2w_rough and config/go2w_rough_wheelpos are only
imported. Both start the curriculum columns on the top row (resume 1000 iters
from the 11-19 ht0003 @13998). leg_energy stays 0 as in that run.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg

from mjlab_playground.velocity.config.go2w_rough.env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg,
)
from mjlab_playground.velocity.config.go2w_rough_wheelpos import wheelpos_mdp
from mjlab_playground.velocity.config.go2w_rough_wheelpos.env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg,
)
from mjlab_playground.velocity.config.go2w_rough_zerostd import zerostd_mdp

ZERO_STD_MIN = 0.1


def _apply_zerostd(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  for name, func in (
    ("track_linear_velocity", zerostd_mdp.track_linear_velocity_terrain_zerostd),
    ("track_angular_velocity", zerostd_mdp.track_angular_velocity_terrain_zerostd),
  ):
    r = cfg.rewards[name]
    assert r.params.get("cmd_scaled") is True, (name, r.params)
    assert r.params.get("std_min") == ZERO_STD_MIN, (name, r.params)
    cfg.rewards[name] = replace(r, func=func)
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _apply_zerostd(unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play))
  if not play:
    cfg.events["init_rough_terrain_at_top"] = EventTermCfg(
      func=wheelpos_mdp.init_rough_terrain_at_top,
      mode="startup",
      params={"skip_terrains": ("flat", "random_rough")},
    )
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  # The WheelPos cfg already adds init_rough_terrain_at_top (not in play).
  return _apply_zerostd(
    unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg(play=play)
  )


# --- 11-24 晚: 从 flat 热启动的完整 rough 训练（不再从已溜车的 13998 续训）---
# 和 ZeroStd-WheelPos 相同，只是去掉 init_rough_terrain_at_top，课程和原 rough run 一样从低排开始。
#   WsFlat:        leg_energy 0，对应 11-19 ht0003（2048 env，warmstart_ht0003）
#   En5e4-WsFlat:  leg_energy -5e-4，对应 11-24 13-24-29 en5e4（4096 env，warmstart_flat_en5e4_n4096）
LEG_ENERGY_EN5E4 = -5e-4


def unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_wsflat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_env_cfg(play=play)
  cfg.events.pop("init_rough_terrain_at_top", None)
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_en5e4_zerostd_wheelpos_wsflat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_wsflat_env_cfg(play=play)
  assert cfg.rewards["leg_energy"].weight == 0.0, cfg.rewards["leg_energy"].weight
  cfg.rewards["leg_energy"].weight = LEG_ENERGY_EN5E4
  return cfg

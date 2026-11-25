"""Go2W rough SA (11-24 晚): Straight-H6-LowNoise-Ht0003 只把 stand_still_wheel_avg
权重放大 (x5 / x10)，针对 0 指令溜车。其他全部同原 task，不加 ZeroStd / WheelPos。

量级: 原 -0.03 时每 episode 约 -0.003，比 action_rate(-0.11)、hip_torques(-0.07)、
heading_drift(-0.03) 小 10~40 倍，站立样本又只占 ~4%，在 PPO 归一化后的优势里被淹没。
从 11-19 ht0003 @13998 续训 1000 iter，课程列从最高排开始（同 ZeroStd/WheelPos 续训实验）。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg

from mjlab_playground.velocity.config.go2w_rough.env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg,
)
from mjlab_playground.velocity.config.go2w_rough_wheelpos import wheelpos_mdp

SA_BASE = -0.03


def _sa_cfg(mult: float, play: bool) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play)
  r = cfg.rewards["stand_still_wheel_avg"]
  assert r.weight == SA_BASE, r.weight
  r.weight = SA_BASE * mult
  if not play:
    cfg.events["init_rough_terrain_at_top"] = EventTermCfg(
      func=wheelpos_mdp.init_rough_terrain_at_top,
      mode="startup",
      params={"skip_terrains": ("flat", "random_rough")},
    )
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_sa5_env_cfg(play: bool = False):
  return _sa_cfg(5.0, play)


def unitree_go2w_rough_straight_h6_lownoise_ht0003_sa10_env_cfg(play: bool = False):
  return _sa_cfg(10.0, play)


# --- Stand25: 平地类地形上 standing 桶 10% -> 25%（从 slow、combo 各拿 7.5%）---
# 其他全同 ht0003（stand_still 权重 -0.03 不变，课程从低排开始），从 flat warmstart_ht0003 完整训练。
STAND25_P = {"standing": 0.25, "vx": 0.10, "vy": 0.10, "wz": 0.10, "slow": 0.225, "combo": 0.225}


def unitree_go2w_rough_straight_h6_lownoise_ht0003_stand25_env_cfg(play: bool = False):
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play)
  tw = cfg.commands["twist"]
  assert tw.p == {"standing": 0.1, "vx": 0.1, "vy": 0.1, "wz": 0.1, "slow": 0.3, "combo": 0.3}, tw.p
  tw.p = dict(STAND25_P)
  return cfg


# --- SaDisp (11-24 晚): 保留 stand_still_wheel_avg(-0.03)，加 0 指令位移惩罚 + critic 位移观测 ---
# 见 disp_mdp.py（锚点在“停下来”后才记，刹车距离不罚）。三个 task：
#   Flat-...-LowNoise-SaDisp : flat 配方（命令行覆盖同 scripts/train_flat_ht0003_en.sh）从零重训，
#                              critic = 原 12 项 + stand_disp
#   Rough-...-Ht0003-SaDisp  : 从上面的 flat 热启动，critic 的 stand_disp 紧跟 dgv2 之后
#                              （warmstart_ckpt.py 按位置映射，旧项必须在最前）
#   Rough-...-Ht0003-SaDisp-Resume : 从 11-19 ht0003 @13998 续训，stand_disp 追加在 critic 最后，
#                              课程列从最高排开始
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_playground.velocity.config.go2w.env_cfgs import (
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg,
)
from mjlab_playground.velocity.config.go2w_rough_sa import disp_mdp

DISP_WEIGHT = -0.5  # 每 10 cm 位移每步 -0.5，封顶 0.2 m（单步最多 -1）。-2/0.5 m 时单步最多 -10 > 全部正奖励，训练中 std 0.51->0.80、总奖励 45->35，已废弃
DISP_SCALE = 0.1
DISP_MAX = 0.2


def _add_disp(cfg: ManagerBasedRlEnvCfg, critic_after: str | None) -> ManagerBasedRlEnvCfg:
  assert "stand_displacement" not in cfg.rewards
  cfg.rewards["stand_displacement"] = RewardTermCfg(
    func=disp_mdp.stand_displacement_penalty,
    weight=DISP_WEIGHT,
    params={"command_name": "twist", "scale": DISP_SCALE, "max_dist": DISP_MAX},
  )
  term = ObservationTermCfg(
    func=disp_mdp.stand_displacement_b,
    params={"command_name": "twist", "max_dist": DISP_MAX},
    scale=1.0 / DISP_MAX,
  )
  terms = cfg.observations["critic"].terms
  if critic_after is None:
    terms["stand_disp"] = term
  else:
    items = list(terms.items())
    i = [k for k, _ in items].index(critic_after) + 1
    terms.clear()
    terms.update(items[:i] + [("stand_disp", term)] + items[i:])
  return cfg


def unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_sadisp_env_cfg(play: bool = False):
  return _add_disp(unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg(play=play), None)


def unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_env_cfg(play: bool = False):
  return _add_disp(unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play), "dgv2")


def unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_resume_env_cfg(play: bool = False):
  cfg = _add_disp(unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play), None)
  if not play:
    cfg.events["init_rough_terrain_at_top"] = EventTermCfg(
      func=wheelpos_mdp.init_rough_terrain_at_top,
      mode="startup",
      params={"skip_terrains": ("flat", "random_rough")},
    )
  return cfg

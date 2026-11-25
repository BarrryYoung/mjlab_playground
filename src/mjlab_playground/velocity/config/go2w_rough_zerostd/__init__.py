from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_playground.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg

from .env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_wsflat_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_en5e4_zerostd_wheelpos_wsflat_env_cfg,
)


def _rl_cfg(name: str):
  cfg = unitree_go2w_ppo_runner_cfg()
  cfg.experiment_name = name
  cfg.wandb_project = name
  return cfg


for task_id, make_cfg, experiment in (
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-ZeroStd-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_zerostd_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-ZeroStd-WheelPos-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-ZeroStd-WheelPos-WsFlat-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_wsflat_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_zerostd_wheelpos_wsflat_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-En5e4-ZeroStd-WheelPos-WsFlat-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_en5e4_zerostd_wheelpos_wsflat_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_en5e4_zerostd_wheelpos_wsflat_velocity",
  ),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_cfg(),
    play_env_cfg=make_cfg(play=True),
    rl_cfg=_rl_cfg(experiment),
    runner_cls=VelocityOnPolicyRunner,
  )

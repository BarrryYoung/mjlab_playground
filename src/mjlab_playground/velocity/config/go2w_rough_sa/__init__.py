from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_playground.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg

from .env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_sa5_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_sa10_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_stand25_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_sadisp_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_resume_env_cfg,
)


def _rl_cfg(name: str):
  cfg = unitree_go2w_ppo_runner_cfg()
  cfg.experiment_name = name
  cfg.wandb_project = name
  return cfg


for task_id, make_cfg, experiment in (
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SA5-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_sa5_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_sa5_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SA10-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_sa10_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_sa10_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Stand25-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_stand25_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_stand25_velocity",
  ),
  (
    "Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise-SaDisp",
    unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_sadisp_env_cfg,
    "playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_sadisp_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SaDisp-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_sadisp_velocity",
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SaDisp-Resume-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_sadisp_resume_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_sadisp_resume_velocity",
  ),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_cfg(),
    play_env_cfg=make_cfg(play=True),
    rl_cfg=_rl_cfg(experiment),
    runner_cls=VelocityOnPolicyRunner,
  )

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_playground.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg

from .env_cfgs import unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg

_EXP = "playground_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_velocity"


def _rl_cfg():
  cfg = unitree_go2w_ppo_runner_cfg()
  cfg.experiment_name = _EXP
  cfg.wandb_project = _EXP
  return cfg


register_mjlab_task(
  task_id="Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-WheelPos-Unitree-Go2W",
  env_cfg=unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg(),
  play_env_cfg=unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg(play=True),
  rl_cfg=_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

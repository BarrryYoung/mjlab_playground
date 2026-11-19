from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2w_rough_env_cfg
from .rl_cfg import unitree_go2w_ppo_runner_cfg

register_mjlab_task(
  task_id="Playground-Velocity-Rough-Unitree-Go2W-Kp40Kd1",
  env_cfg=unitree_go2w_rough_env_cfg(),
  play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

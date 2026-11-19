from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2w_flat_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_clean50_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_env_cfg,
  unitree_go2w_flat_isaacobs_cmd_m22_env_cfg,
  unitree_go2w_flat_isaacobs_env_cfg,
  unitree_go2w_flat_isaacobs_yaw2_env_cfg,
  unitree_go2w_flat_isaacobs_yaw2_fb_env_cfg,
  unitree_go2w_flat_isaacobs_yaw2_fb_low_env_cfg,
  unitree_go2w_flat_isaacobs_yaw2_fb_low_m22_env_cfg,
  unitree_go2w_rough_env_cfg,
)
from .rl_cfg import unitree_go2w_ppo_runner_cfg

_flat_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_rl_cfg.experiment_name = "playground_go2w_flat_velocity"
_flat_rl_cfg.wandb_project = "playground_go2w_flat_velocity"

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W",
  env_cfg=unitree_go2w_flat_env_cfg(),
  play_env_cfg=unitree_go2w_flat_env_cfg(play=True),
  rl_cfg=_flat_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_rl_cfg.experiment_name = "playground_go2w_flat_isaacobs_velocity"
_flat_isaacobs_rl_cfg.wandb_project = "playground_go2w_flat_isaacobs_velocity"

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs",
  env_cfg=unitree_go2w_flat_isaacobs_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_rl_cfg.experiment_name = "playground_go2w_flat_isaacobs_cmd_velocity"
_flat_isaacobs_cmd_rl_cfg.wandb_project = "playground_go2w_flat_isaacobs_cmd_velocity"

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_dgv2_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_dgv2_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_velocity"
)
_flat_isaacobs_cmd_dgv2_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_dgv2_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_dgv2_clean50_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_dgv2_clean50_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_clean50_velocity"
)
_flat_isaacobs_cmd_dgv2_clean50_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_clean50_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-Clean50",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_clean50_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_clean50_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_dgv2_clean50_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_dgv2_m22_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_dgv2_m22_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_velocity"
)
_flat_isaacobs_cmd_dgv2_m22_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_dgv2_m22_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_dgv2_m22_lownoise_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_dgv2_m22_lownoise_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_velocity"
)
_flat_isaacobs_cmd_dgv2_m22_lownoise_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_dgv2_m22_lownoise_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_velocity"
)
_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise-HullWheel",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg(
    play=True
  ),
  rl_cfg=_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_cmd_m22_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_cmd_m22_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_cmd_m22_velocity"
)
_flat_isaacobs_cmd_m22_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_cmd_m22_velocity"
)
_flat_isaacobs_cmd_m22_rl_cfg.max_iterations = 4000

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-M22",
  env_cfg=unitree_go2w_flat_isaacobs_cmd_m22_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_cmd_m22_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_cmd_m22_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_yaw2_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_yaw2_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_yaw2_velocity"
)
_flat_isaacobs_yaw2_rl_cfg.wandb_project = "playground_go2w_flat_isaacobs_yaw2_velocity"

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Yaw2",
  env_cfg=unitree_go2w_flat_isaacobs_yaw2_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_yaw2_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_yaw2_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_yaw2_fb_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_yaw2_fb_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_yaw2_fb_velocity"
)
_flat_isaacobs_yaw2_fb_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_yaw2_fb_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Yaw2-FB",
  env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_yaw2_fb_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_yaw2_fb_low_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_yaw2_fb_low_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_yaw2_fb_low_velocity"
)
_flat_isaacobs_yaw2_fb_low_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_yaw2_fb_low_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Yaw2-FB-Low",
  env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_low_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_low_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_yaw2_fb_low_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_flat_isaacobs_yaw2_fb_low_m22_rl_cfg = unitree_go2w_ppo_runner_cfg()
_flat_isaacobs_yaw2_fb_low_m22_rl_cfg.experiment_name = (
  "playground_go2w_flat_isaacobs_yaw2_fb_low_m22_velocity"
)
_flat_isaacobs_yaw2_fb_low_m22_rl_cfg.wandb_project = (
  "playground_go2w_flat_isaacobs_yaw2_fb_low_m22_velocity"
)

register_mjlab_task(
  task_id="Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Yaw2-FB-Low-M22",
  env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_low_m22_env_cfg(),
  play_env_cfg=unitree_go2w_flat_isaacobs_yaw2_fb_low_m22_env_cfg(play=True),
  rl_cfg=_flat_isaacobs_yaw2_fb_low_m22_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Playground-Velocity-Rough-Unitree-Go2W",
  env_cfg=unitree_go2w_rough_env_cfg(),
  play_env_cfg=unitree_go2w_rough_env_cfg(play=True),
  rl_cfg=unitree_go2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_playground.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg

from .env_cfgs import (
  unitree_go2w_climb_blind_dc_env_cfg,
  unitree_go2w_climb_blind_dc_h65_env_cfg,
  unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg,
  unitree_go2w_climb_blind_dc_h65_flatpose_bar2_stop_env_cfg,
  unitree_go2w_climb_blind_dc_h65_flatpose_bar_env_cfg,
  unitree_go2w_climb_blind_dc_h65_flatpose_env_cfg,
  unitree_go2w_climb_blind_dc_h65_flatpose_trunkterm_env_cfg,
  unitree_go2w_climb_blind_env_cfg,
  unitree_go2w_climb_blind_h65_env_cfg,
  unitree_go2w_climb_blind_nobit_dc_h65_env_cfg,
  unitree_go2w_climb_dc_h65_flatpose_bar2_env_cfg,
  unitree_go2w_climb_dc_h65_flatpose_bar_env_cfg,
  unitree_go2w_climb_env_cfg,
  unitree_go2w_climb_h65_env_cfg,
)


def _rl_cfg(name: str):
  cfg = unitree_go2w_ppo_runner_cfg()
  cfg.experiment_name = name
  cfg.wandb_project = name
  return cfg


# The H65 variants log under the same experiment as their 0.50 m parent, so
# `--agent.resume True --agent.load-run <parent run>` finds the parent's
# checkpoints (resume looks under logs/rsl_rl/<experiment_name>/).
_TASKS = (
  (
    "Playground-Velocity-Climb-Unitree-Go2W",
    unitree_go2w_climb_env_cfg,
    "playground_go2w_climb_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-Unitree-Go2W",
    unitree_go2w_climb_blind_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-H65-Unitree-Go2W",
    unitree_go2w_climb_h65_env_cfg,
    "playground_go2w_climb_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-H65-Unitree-Go2W",
    unitree_go2w_climb_blind_h65_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_flatpose_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-TrunkTerm-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_flatpose_trunkterm_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_flatpose_bar_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-DC-H65-FlatPose-Bar-Unitree-Go2W",
    unitree_go2w_climb_dc_h65_flatpose_bar_env_cfg,
    "playground_go2w_climb_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-DC-H65-FlatPose-Bar2-Unitree-Go2W",
    unitree_go2w_climb_dc_h65_flatpose_bar2_env_cfg,
    "playground_go2w_climb_velocity",
  ),
  (
    "Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Stop-Unitree-Go2W",
    unitree_go2w_climb_blind_dc_h65_flatpose_bar2_stop_env_cfg,
    "playground_go2w_climb_blind_velocity",
  ),
  (
    "Playground-Velocity-Climb-BlindNoBit-DC-H65-Unitree-Go2W",
    unitree_go2w_climb_blind_nobit_dc_h65_env_cfg,
    "playground_go2w_climb_blind_nobit_velocity",
  ),
)

for task_id, make_cfg, experiment in _TASKS:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_cfg(),
    play_env_cfg=make_cfg(play=True),
    rl_cfg=_rl_cfg(experiment),
    runner_cls=VelocityOnPolicyRunner,
  )

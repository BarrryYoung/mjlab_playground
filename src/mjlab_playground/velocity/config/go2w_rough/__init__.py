from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_playground.velocity.config.go2w.rl_cfg import unitree_go2w_ppo_runner_cfg

from .env_cfgs import (
  unitree_go2w_rough_dgv2_m22_env_cfg,
  unitree_go2w_rough_dgv2_m22_track_l0_env_cfg,
  unitree_go2w_rough_dgv2_m22_track_l0_wz_env_cfg,
  unitree_go2w_rough_dgv2_m22_track_l0_wz_h6_env_cfg,
  unitree_go2w_rough_heading_fwd_h6_env_cfg,
  unitree_go2w_rough_heading_h6_env_cfg,
  unitree_go2w_rough_straight_h6_env_cfg,
  unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg,
)


def _rl_cfg(name: str):
  cfg = unitree_go2w_ppo_runner_cfg()
  cfg.experiment_name = name
  cfg.wandb_project = name
  return cfg


def _gru_rl_cfg(name: str):
  """Actor: obs -> GRU(128) -> h; MLP(512-256-128) on [obs, h] (see SkipRNNModel).
  The critic stays an MLP. Truncated BPTT over each 24-step rollout."""
  cfg = _rl_cfg(name)
  cfg.actor = replace(
    cfg.actor,
    class_name="mjlab_playground.rl.skip_rnn:SkipRNNModel",
    rnn_type="gru",
    rnn_hidden_dim=128,
    rnn_num_layers=1,
  )
  return cfg


for task_id, make_cfg, experiment, make_rl in (
  (
    "Playground-Velocity-Rough-Dgv2-M22-Unitree-Go2W",
    unitree_go2w_rough_dgv2_m22_env_cfg,
    "playground_go2w_rough_dgv2_m22_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Dgv2-M22-TrackL0-Unitree-Go2W",
    unitree_go2w_rough_dgv2_m22_track_l0_env_cfg,
    "playground_go2w_rough_dgv2_m22_trackl0_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Dgv2-M22-TrackL0-Wz-Unitree-Go2W",
    unitree_go2w_rough_dgv2_m22_track_l0_wz_env_cfg,
    "playground_go2w_rough_dgv2_m22_trackl0wz_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Dgv2-M22-TrackL0-Wz-H6-Unitree-Go2W",
    unitree_go2w_rough_dgv2_m22_track_l0_wz_h6_env_cfg,
    "playground_go2w_rough_dgv2_m22_trackl0wzh6_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Dgv2-M22-TrackL0-Wz-Gru-Unitree-Go2W",
    unitree_go2w_rough_dgv2_m22_track_l0_wz_env_cfg,
    "playground_go2w_rough_dgv2_m22_trackl0wzgru_velocity",
    _gru_rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Heading-H6-Unitree-Go2W",
    unitree_go2w_rough_heading_h6_env_cfg,
    "playground_go2w_rough_heading_h6_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Heading-Fwd-H6-Unitree-Go2W",
    unitree_go2w_rough_heading_fwd_h6_env_cfg,
    "playground_go2w_rough_heading_fwd_h6_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_env_cfg,
    "playground_go2w_rough_straight_h6_velocity",
    _rl_cfg,
  ),
  (
    "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Unitree-Go2W",
    unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg,
    "playground_go2w_rough_straight_h6_lownoise_ht0003_velocity",
    _rl_cfg,
  ),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_cfg(),
    play_env_cfg=make_cfg(play=True),
    rl_cfg=make_rl(experiment),
    runner_cls=VelocityOnPolicyRunner,
  )

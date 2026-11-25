"""Go2W rough WheelPos: Straight-H6-LowNoise-Ht0003 + a wheel speed from the
wheel angle in the actor (11-24, aimed at the policy creeping on a zero command).

New task family; nothing in config/go2w_rough is modified, only imported.

  - Actor: the 342 inputs unchanged (6 frames each, same order), plus
    wheel_vel_pos (4, single frame) appended last: (q_t - q_{t-5}) / 0.1 s per
    wheel, FR FL RR RL, scale 0.05 as joint_vel, noise +-0.02 rad/s. 346 inputs.
    tools/warmstart_ckpt.py keeps the old 342 columns and zeroes the new 4.
  - Startup: the curriculum columns start on the top row (resume, 1000 iters).
  - Everything else (rewards, noise, commands, terrain, robot, critic) as
    Straight-H6-LowNoise-Ht0003; leg_energy stays 0 as in its 11-19 run.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.velocity.config.go2w_rough.env_cfgs import (
  unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg,
)
from mjlab_playground.velocity.config.go2w_rough_wheelpos import wheelpos_mdp

WHEEL_JOINTS = ("FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint")
WHEEL_POS_WINDOW = 5  # policy steps (0.1 s)


def unitree_go2w_rough_straight_h6_lownoise_ht0003_wheelpos_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play)
  actor = cfg.observations["actor"]
  hist, flat = actor.history_length, actor.flatten_history_dim
  assert hist == 6 and flat, (hist, flat)
  # Group history would also stack the new term; give each old term its own 6
  # frames instead (same layout) and leave the new one at a single frame.
  actor.history_length = None
  for name in list(actor.terms):
    actor.terms[name] = replace(
      actor.terms[name], history_length=hist, flatten_history_dim=flat
    )
  actor.terms["wheel_vel_pos"] = ObservationTermCfg(
    func=wheelpos_mdp.wheel_vel_from_pos,
    params={"joint_names": WHEEL_JOINTS, "window": WHEEL_POS_WINDOW},
    noise=Unoise(n_min=-0.02, n_max=0.02),
    scale=actor.terms["joint_vel"].scale,
    clip=actor.terms["joint_vel"].clip,
  )
  if not play:
    cfg.events["init_rough_terrain_at_top"] = EventTermCfg(
      func=wheelpos_mdp.init_rough_terrain_at_top,
      mode="startup",
      params={"skip_terrains": ("flat", "random_rough")},
    )
  return cfg

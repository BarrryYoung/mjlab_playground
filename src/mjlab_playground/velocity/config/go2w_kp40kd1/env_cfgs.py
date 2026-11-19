"""Unitree Go2W velocity environment configurations -- leg PD kp=40 kd=1.

Copy of `config/go2w/env_cfgs.py`; the only change is the leg actuator gains
(25/0.5 -> 40/1, DreamWaQ_Go2W's training value), marked `# Kp40Kd1:` below.

Mirrors `config/go1/env_cfgs.py` (rough). Deviations from Go1 are the ones a
wheeled-legged robot forces, each marked `# Go2W:` below:
  - actions split into leg joint positions + wheel joint velocities;
  - wheel joints excluded from joint_pos obs (unbounded angle), from
    joint_pos_limits (wheel jnt_range is [0, 0] = "unlimited", which the reward
    reads as a hard limit at 0) and from the posture kernel;
  - foot_slip / foot_clearance / foot_swing_height off: wheels roll while in
    contact and need not lift.
"""

import math
from dataclasses import dataclass, fields, replace

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg

from mjlab_playground.asset_zoo.robots.unitree_go2w.go2w_constants import (
  BASE_BODY,
  FOOT_NAMES,
  GO2W_LEG_ACTUATOR_CFG,
  GO2W_WHEEL_ACTUATOR_CFG,
  WHEEL_GEOM_NAMES,
  WHEEL_SITE_NAMES,
  get_go2w_robot_cfg,
)
from mjlab_playground.velocity import mdp
from mjlab_playground.velocity.velocity_env_cfg import make_velocity_env_cfg

_LEG_JOINTS = (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
_WHEEL_JOINTS = (".*_wheel_joint",)

# Go2W: robot_lab action scales (legs 0.25 rad, wheels 5.0 rad/s).
_LEG_ACTION_SCALE = 0.25
_WHEEL_ACTION_SCALE = 5.0


##
# Rewards
##


def _go2w_rewards() -> dict[str, RewardTermCfg]:
  """Go1's reward set with wheel-incompatible terms switched off."""
  return {
    # -- Task --------------------------------------------------------------
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # -- Posture -----------------------------------------------------------
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),  # Go2W
        "command_name": "twist",
        "std_standing": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
        },
        "std_walking": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        "std_running": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # -- Regularization ----------------------------------------------------
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},  # Go2W
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=0.0,  # Go2W: wheels roll in contact.
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=WHEEL_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Off by default.
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Off by default.
      params={"sensor_name": "robot/root_angmom"},
    ),
    # -- Shaping -----------------------------------------------------------
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=0.0,  # Go2W: no forced stepping.
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=WHEEL_SITE_NAMES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=0.0,  # Go2W: no forced stepping.
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Off by default.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
  }


##
# Robot wiring
##


@dataclass
class Go2WMujocoCfg(MujocoCfg):
  """MujocoCfg plus `ccd_tolerance`, which mjlab's MujocoCfg does not expose."""

  # Go2W: MuJoCo's default 1e-6 is finer than float32 resolution at the env
  # origins (~4e-6 m at x≈38 m); wheel-cylinder vs stair-box EPA then picks a
  # degenerate face and puts the contact tens of meters away -> NaN
  # (mujoco_warp#1782).
  ccd_tolerance: float = 1e-5

  def apply(self, model: mujoco.MjModel) -> None:
    super().apply(model)
    model.opt.ccd_tolerance = self.ccd_tolerance


def _make_go2w_base_cfg() -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco = Go2WMujocoCfg(
    **{f.name: getattr(cfg.sim.mujoco, f.name) for f in fields(MujocoCfg)}
  )
  # Go2W: put_data checks nconmax against a CPU mj_forward of the pre-reset
  # state (robot at the world origin, z=0.45). With play's random terrain the
  # origin can sit under a hill and bury the whole robot: up to 76 contacts in
  # 40 draws, vs Go1's 35.
  cfg.sim.nconmax = 128

  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"

  # Kp40Kd1: stiffer legs; wheels, effort limit and everything else unchanged.
  robot = get_go2w_robot_cfg()
  robot.articulation = replace(
    robot.articulation,
    actuators=(
      replace(GO2W_LEG_ACTUATOR_CFG, stiffness=40.0, damping=1.0),
      GO2W_WHEEL_ACTUATOR_CFG,
    ),
  )
  cfg.scene.entities = {"robot": robot}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in WHEEL_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=WHEEL_GEOM_NAMES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
  )

  # Go2W: legs position-controlled, wheels velocity-controlled.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_LEG_JOINTS,
      scale=_LEG_ACTION_SCALE,
      use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_WHEEL_JOINTS,
      scale=_WHEEL_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # Go2W: wheel angle is unbounded, keep only leg positions in obs.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=_LEG_JOINTS
    )

  cfg.viewer.body_name = BASE_BODY
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  return cfg


##
# Rough terrain
##


def unitree_go2w_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2W rough terrain velocity configuration."""
  cfg = _make_go2w_base_cfg()
  cfg.rewards = _go2w_rewards()
  cfg.metrics["track_lin_vel_err"] = MetricsTermCfg(
    func=mdp.track_lin_vel_err, params={"command_name": "twist"}
  )
  cfg.metrics["track_yaw_rate_err"] = MetricsTermCfg(
    func=mdp.track_yaw_rate_err, params={"command_name": "twist"}
  )

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)

  # Geom names as in xmls/go2w.xml. Self collisions are disabled in the model
  # (conaffinity=0), so self_collision stays at 0; kept to mirror Go1.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=tuple(f"{leg}_thigh_collision" for leg in FOOT_NAMES),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=tuple(f"{leg}_calf{i}_collision" for leg in FOOT_NAMES for i in (1, 2)),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=tuple(f"base{i}_collision" for i in (1, 2, 3)),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Go2W: keep fell_over (Go1 drops it) at 60 deg, as in the Isaac Go2W setup;
  # steepest terrain is ~37.5 deg.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(60.0)},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg

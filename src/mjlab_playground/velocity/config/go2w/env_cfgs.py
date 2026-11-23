"""Unitree Go2W velocity environment configurations.

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
from mjlab.managers.observation_manager import ObservationTermCfg
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
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.asset_zoo.robots.unitree_go2w.go2w_constants import (
  ALIGNED_COLLISION,
  ALIGNED_V2_COLLISION,
  BASE_BODY,
  FOOT_NAMES,
  GO2W_ALIGNED_ARTICULATION,
  WHEEL_GEOM_NAMES,
  WHEEL_SITE_NAMES,
  get_go2w_robot_cfg,
  get_spec,
  get_spec_m22,
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

  cfg.scene.entities = {"robot": get_go2w_robot_cfg()}

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
# Flat terrain
##

# Go1's foot site sits at the foot-sphere center (r=0.023), so its 0.1 m target
# means ~7.7 cm of lift. The wheel site is at the wheel center (r=0.086); same lift.
_GO2W_SWING_TARGET_HEIGHT = 0.086 + (0.1 - 0.023)


def unitree_go2w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2W on a plane with Go1's full reward set (stepping terms back on)."""
  cfg = _make_go2w_base_cfg()
  cfg.rewards = _go2w_rewards()
  # Go1 weights for the terms the rough Go2W task switches off. Note foot_slip
  # reads the wheel-center velocity, so it also penalizes rolling.
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_clearance"].params["target_height"] = _GO2W_SWING_TARGET_HEIGHT
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = _GO2W_SWING_TARGET_HEIGHT
  cfg.metrics["track_lin_vel_err"] = MetricsTermCfg(
    func=mdp.track_lin_vel_err, params={"command_name": "twist"}
  )
  cfg.metrics["track_yaw_rate_err"] = MetricsTermCfg(
    func=mdp.track_yaw_rate_err, params={"command_name": "twist"}
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan on a plane.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(60.0)},
  )
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg


##
# Flat terrain, robot_lab (Isaac) actor interface
##

# robot_lab / unitree SDK motor order: legs FR, FL, RR, RL then wheels.
_SDK_LEGS = tuple(
  f"{leg}_{j}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for j in ("hip", "thigh", "calf")
)
_SDK_WHEELS = tuple(f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL"))
_ISAAC_CLIP = (-100.0, 100.0)


def unitree_go2w_flat_isaacobs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat task with the actor obs/action layout of robot_lab's Go2W (57 -> 16).

  Actor obs, in order: base_ang_vel x0.25, projected_gravity, command,
  joint_pos_rel_without_wheel (16), joint_vel x0.05 (16), last_action (16);
  joints in SDK order, clip +-100, same noise as robot_lab. No base_lin_vel.
  Actions are SDK-ordered too. Critic, rewards and everything else are the flat
  task's. Leg action scale stays 0.25 (robot_lab uses 0.125 on hips).
  """
  cfg = unitree_go2w_flat_env_cfg(play=play)

  sdk_joints = SceneEntityCfg(
    "robot", joint_names=_SDK_LEGS + _SDK_WHEELS, preserve_order=True
  )
  # New term objects: the critic dict shares the actor's ObservationTermCfg
  # instances, so mutating them in place would change the critic too.
  cfg.observations["actor"].terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      clip=_ISAAC_CLIP,
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      clip=_ISAAC_CLIP,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
      clip=_ISAAC_CLIP,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel_without_wheel,
      params={"asset_cfg": sdk_joints},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      clip=_ISAAC_CLIP,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": replace(sdk_joints)},
      noise=Unoise(n_min=-1.5, n_max=1.5),
      clip=_ISAAC_CLIP,
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, clip=_ISAAC_CLIP),
  }

  # mjlab's own joint actions ignore preserve_order; the Ordered* ones honor it.
  cfg.actions = {
    "joint_pos": mdp.OrderedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=_SDK_LEGS,
      preserve_order=True,
      scale=_LEG_ACTION_SCALE,
      use_default_offset=True,
    ),
    "wheel_vel": mdp.OrderedJointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_SDK_WHEELS,
      preserve_order=True,
      scale=_WHEEL_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # Nothing else stops the wheels under a zero command: a constant wheel action
  # is a constant velocity target, which action_rate does not see. L2 at -0.01
  # (the "still" run) brought the deterministic standing wheel speed from 0.44
  # to 0.05 rad/s with leg std 0.19 -> 0.22 and tracking unchanged. The training
  # curve barely moves because ~97% of the sampled sum w^2 is exploration noise
  # (wheel std ~0.6 x scale 5); judge it with the noise off. At -1.0 that noise
  # floor dominates and the policy learned to fall over. Off by default, set the
  # weight per run.
  cfg.rewards["stand_still_wheel_vel"] = RewardTermCfg(
    func=mdp.stand_still_wheel_vel_l2,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
    },
  )
  # Same penalty on the ~0.5 s average wheel speed: the per-step exploration
  # noise averages out, the creeping bias does not. From scratch the L2 above
  # stalled learning (s001: leg std stuck at 0.8, sat down when standing).
  cfg.rewards["stand_still_wheel_avg"] = RewardTermCfg(
    func=mdp.stand_still_wheel_vel_avg,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
      "tau": 0.5,
    },
  )
  # Same average, L1 kernel: L2's pull (2w) fades near zero and sa003 drifted
  # 0.08-0.28 rad/s late on. At -0.03 the from-scratch noise floor costs about
  # what L2 -0.03 does (sum|avg| ~2.6 vs sum avg^2 ~2.7); at 0.1 rad/s the pull
  # is 5x stronger.
  cfg.rewards["stand_still_wheel_avg_l1"] = RewardTermCfg(
    func=mdp.stand_still_wheel_vel_avg,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
      "tau": 0.5,
      "kernel": "l1",
    },
  )
  # Same, but the gate also waits for ema(|command|) to settle: a grace period
  # for braking that scales with the speed being dropped (0.2 s from 1 m/s).
  cfg.rewards["stand_still_wheel_avg_cmd_ema"] = RewardTermCfg(
    func=mdp.stand_still_wheel_vel_avg_cmd_ema,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
      "tau": 0.5,
      "cmd_tau": 0.067,  # 0.2 / ln(1.0 / 0.05)
    },
  )
  # Bounded exp alternative, off by default: std is sized to the *sampled* wheel
  # speed (sum w^2 ~ 38; at std 1 the kernel is ~0.002, no signal). At w 2 std 5
  # it stopped the wheels less (0.18 rad/s) and raised leg std 0.19 -> 0.33.
  cfg.rewards["stand_still_wheel"] = RewardTermCfg(
    func=mdp.stand_still_wheel_vel_exp,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
      "std": 5.0,
    },
  )
  # Replaces foot_swing_height: that term measures the wheel center (a wheel that
  # never lifts still reads 0.086 m) and pairs contact/height sensors in
  # different wheel orders. This one uses clearance and realigns the sensors.
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["foot_swing_clearance"] = RewardTermCfg(
    func=mdp.feet_swing_clearance,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_clearance": 0.04,
      "command_name": "twist",
      "command_threshold": 0.1,  # on |vy| + |wz|
    },
  )
  # Upstream foot_swing_height with only the sensor order fixed (center height,
  # full-command gate), to isolate what the misalignment cost swing1. Off by
  # default.
  cfg.rewards["foot_swing_height_aligned"] = RewardTermCfg(
    func=mdp.feet_swing_height_aligned,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.126,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  # Constant alive bonus, to tell apart upright's posture shaping from the
  # survival reward it mostly acts as. Off by default.
  cfg.rewards["is_alive"] = RewardTermCfg(func=mdp.is_alive, weight=0.0)
  # Even load when driving straight / standing, an even diagonal trot when
  # steering (see diagonal_gait). The policy rode the FR/RL diagonal with FL/RR
  # carrying 1-8% of the load, and stepped with one diagonal while the other
  # jittered. Off by default.
  cfg.rewards["diagonal_gait"] = RewardTermCfg(
    func=mdp.diagonal_gait,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "target_air": 0.15,  # s, swing length the gate is centred on (normal 0.12-0.16 s)
      "air_sigma": 0.05,  # s, q = exp(-((T - target_air) / air_sigma)^2)
      "straight_threshold": 1e-3,  # |vy| + |wz|: vx only / standing
      "step_threshold": 0.1,  # |vy| + |wz|: same gate as foot_swing_clearance
    },
  )

  return cfg


_YAW2_RANGE = (-2.0, 2.0)


def unitree_go2w_flat_isaacobs_yaw2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """IsaacObs task with the yaw-rate command widened to +-2 rad/s.

  Heading envs are clipped to the same range, and every command_vel curriculum
  stage is pinned to it so the stage at 5000 iters cannot shrink it back.
  """
  cfg = unitree_go2w_flat_isaacobs_env_cfg(play=play)
  cfg.commands["twist"].ranges.ang_vel_z = _YAW2_RANGE
  if "command_vel" in cfg.curriculum:
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {**s, "ang_vel_z": _YAW2_RANGE} for s in stages
    ]
  return cfg


def unitree_go2w_flat_isaacobs_yaw2_fb_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Yaw2 task whose forward-only envs (20%) go forward or backward, half each.

  Also rewards all four wheels on the ground when driving straight or standing:
  the policy rode on the FR/RL diagonal, RR in contact ~half the time and FL
  ~20% when reversing. At +0.5 all four reach 100% contact (g05 run).
  """
  cfg = unitree_go2w_flat_isaacobs_yaw2_env_cfg(play=play)
  twist = cfg.commands["twist"]
  cfg.commands["twist"] = mdp.SignedForwardVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}
  )
  cfg.rewards["wheels_grounded"] = RewardTermCfg(
    func=mdp.wheels_grounded,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,  # on |vy| + |wz|, same as foot_swing_clearance
    },
  )
  return cfg


def unitree_go2w_flat_isaacobs_yaw2_fb_low_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FB task with mutually exclusive command groups plus a low-speed group.

  standing 10 / heading 22 / forward (fwd or back) 18 / low 15 / uniform 35.
  The low group (|vx|,|vy| <= 0.3, |wz| <= 0.4, each zeroed w.p. 0.5) is cut
  from the uniform one: with wz ~ U(+-2) slow turns were almost never sampled,
  and at wz = 0.05 the fb policies turned at 0.07-0.12 with a twisted stance.
  The grounded / swing gate on |vy| + |wz| moves from 0.1 to 0.05, the smallest
  command used in practice: at 0.1, wz = 0.05 fell on the grounded side, the
  wheels were held down while the body turned, and the legs twisted up. Now
  anything >= 0.05 is on the swing side (step to adjust), below it the wheels
  stay down. 0.05 also matches the pose term's walking_threshold.
  """
  cfg = unitree_go2w_flat_isaacobs_yaw2_fb_env_cfg(play=play)
  twist = cfg.commands["twist"]
  cfg.commands["twist"] = mdp.GroupedVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}
  )
  cfg.rewards["wheels_grounded"].params["command_threshold"] = 0.05
  cfg.rewards["foot_swing_clearance"].params["command_threshold"] = 0.05
  return cfg


def unitree_go2w_flat_isaacobs_cmd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """IsaacObs task with a redesigned command distribution, nothing else changed.

  Upstream rolls standing / heading / forward as independent dice over
  vx, vy, wz uniform, so steering-only and slow commands are rare: |vy| + |wz|
  in (0.1, 0.3] was 5% of samples and a lone wz 0.15 about 1%. Every policy
  failed wz 0.15 in the gait benchmark, the cleanest one included. Here each
  resample lands in exactly one bucket (see BucketVelocityCommand):

    standing 10 / vx only 10 / vy only 10 / wz only 10 (|v| from 0.3 up) /
    slow single-axis 30 (|v| 0.05-0.3) / combo uniform 30

  Ranges vx +-1, vy +-0.5, wz +-2; the command_vel curriculum is dropped so
  they stay fixed. Reward gates are left as in IsaacObs (swing 0.1, standing
  0.05) to keep the command distribution the only variable.
  """
  cfg = unitree_go2w_flat_isaacobs_env_cfg(play=play)
  twist = cfg.commands["twist"]
  kwargs = {f.name: getattr(twist, f.name) for f in fields(twist)}
  kwargs.update(
    heading_command=False,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-2.0, 2.0), heading=None
    ),
  )
  cfg.commands["twist"] = mdp.BucketVelocityCommandCfg(**kwargs)
  cfg.curriculum.pop("command_vel", None)
  return cfg


def unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Cmd task with the window-based diagonal trot term dgv2 (diagonal_gait_v2).

  - dgv2: load balance when straight / standing, else a trot of period 0.5 s
    scored on the last period. Off by default like every other term here.
  - Command deadzone 0.05 per axis, so |vy| + |wz| is 0 or >= 0.05 and dgv2 has
    no gap between its two cases; the robot side needs the same deadzone.
  - The critic also sees dgv2's window summary (5 values); the actor and so the
    deployed policy are unchanged.
  """
  cfg = unitree_go2w_flat_isaacobs_cmd_env_cfg(play=play)
  cfg.commands["twist"] = replace(cfg.commands["twist"], deadzone=0.05)
  cfg.rewards["dgv2"] = RewardTermCfg(
    func=mdp.diagonal_gait_v2,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "period": 0.5,  # s, window and gait period; each diagonal down for half
      "sigma": 0.05,  # s, q = exp(-((x - target) / sigma)^2)
      "contact_threshold": 0.8,  # c1 * c2 for a diagonal to count as down
      "full_load_frac": 0.5,  # c = 1 from half a fair share of the load
      "straight_threshold": 1e-3,  # |vy| + |wz|: vx only / standing
      "balance_power": 1.0,  # straight score = balance ** balance_power
      "require_all_down": False,  # straight score x min(c): 0 if a wheel is up
    },
  )
  cfg.observations["critic"].terms["dgv2"] = ObservationTermCfg(
    func=mdp.dgv2_state, params={"reward_name": "dgv2"}
  )
  # sum of squared leg actuator torques (N·m)^2; wheels left out. Off by default.
  # dgv2w05 @4100 averages ~540 on legs, ~2.6 on wheels.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=0.0,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=_LEG_JOINTS)},
  )
  # Same on the 4 hips only (they run hot on the real robot), and only when
  # |vy| + |wz| < 0.05 (vx only / standing). Off by default.
  cfg.rewards["hip_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2_straight,
    weight=0.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*_hip_joint",)),
      "step_threshold": 0.05,
    },
  )
  # Electrical power of the 12 leg motors in W: sum(max(tau * qd, 0) + c_j * tau^2),
  # c_j fitted on the real robot (battery power vs tau_est, 11-10 log). Copper
  # loss dominates: standing at the default pose is ~160 W in sim. Off by default.
  cfg.rewards["leg_energy"] = RewardTermCfg(
    func=mdp.leg_electrical_power,
    weight=0.0,
    params={
      "coeffs": {".*_hip_joint": 0.53, ".*_thigh_joint": 0.60, ".*_calf_joint": 0.17},
    },
  )
  # Same tracking terms with an opt-in command-scaled std (params cmd_scaled,
  # std_min); off by default, identical to mjlab's then.
  for name, func in (
    ("track_linear_velocity", mdp.track_linear_velocity_scaled),
    ("track_angular_velocity", mdp.track_angular_velocity_scaled),
  ):
    term = cfg.rewards[name]
    cfg.rewards[name] = replace(
      term, func=func, params={**term.params, "cmd_scaled": False, "std_min": 0.05}
    )
  return cfg


def unitree_go2w_flat_isaacobs_cmd_dgv2_clean50_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dgv2 task with the actor's observation noise off in half of the envs.

  With the default noise (joint_vel +-1.5, ang_vel +-0.2, ...) the trot that
  keeps going after a small wz / vy command drops to zero is broken by the
  noise in training, so it is never penalized, while the deployed policy sees
  clean observations and keeps trotting. The first clean_frac of the envs (by
  index) get no noise; the rest keep the same ranges as Dgv2.
  """
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg(play=play)
  terms = cfg.observations["actor"].terms
  for name, term in terms.items():
    if isinstance(term.noise, Unoise):
      noise = mdp.EnvSplitUniformNoiseCfg(
        n_min=term.noise.n_min, n_max=term.noise.n_max, clean_frac=0.5
      )
      terms[name] = replace(term, noise=noise)
  return cfg


def unitree_go2w_flat_isaacobs_cmd_m22_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Cmd task at the real mass: xmls/go2w_m22.xml (22.0 kg), +-1 kg on the base.

  The mass lives in the MJCF rather than a spec_fn edit, so sim2sim can load the
  same file. The +-1 kg is a point mass at the base COM (dr.body_mass leaves
  inertia alone), as in the Yaw2-FB-Low-M22 task. Rewards are the Cmd task's.
  """
  cfg = unitree_go2w_flat_isaacobs_cmd_env_cfg(play=play)
  cfg.scene.entities["robot"] = replace(
    cfg.scene.entities["robot"], spec_fn=get_spec_m22
  )
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      "operation": "add",
      "ranges": (-1.0, 1.0),
    },
  )
  return cfg


def unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dgv2 task at the real mass: same robot and +-1 kg as Cmd-M22, rest is Dgv2's."""
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_env_cfg(play=play)
  m22 = unitree_go2w_flat_isaacobs_cmd_m22_env_cfg(play=play)
  cfg.scene.entities["robot"] = m22.scene.entities["robot"]
  cfg.events["base_mass"] = m22.events["base_mass"]
  return cfg


def unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dgv2-M22 with smaller actor noise on ang_vel and joint_vel.

  The legged_gym defaults (ang_vel +-0.2, joint_vel +-1.5) were measured on
  ANYmal C. On the real Go2-W while walking we measured about +-0.09 and
  +-0.47 (both upper bounds), so: ang_vel +-0.05, joint_vel +-0.5.
  projected_gravity and joint_pos keep their defaults.
  """
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg(play=play)
  terms = cfg.observations["actor"].terms
  for name, a in (("base_ang_vel", 0.05), ("joint_vel", 0.5)):
    assert isinstance(terms[name].noise, Unoise), name
    terms[name] = replace(terms[name], noise=Unoise(n_min=-a, n_max=a))
  return cfg


_WHEEL_HULL_MAXVERT = (
  -1
)  # full hull (786 verts): rolling radius varies 0.18 mm; 64 -> 5 mm, 256 -> 2.6 mm


def get_spec_m22_hull_wheels() -> mujoco.MjSpec:
  """go2w_m22.xml with each wheel collision geom swapped from a cylinder to wheel.stl.

  MuJoCo collides a mesh through its convex hull, so the wheel gets the real tyre
  profile: r 8.6 cm at the centre rounding off to 7.0 cm at the sidewalls, 5.2 cm
  wide, where the cylinder is r 8.6 cm and 4.5 cm wide with square edges that
  catch on step noses. The geom takes the visual geom's pose (the mesh frame), so
  its names and the contact sensors are unchanged. The hull is left uncapped
  (786 of the mesh's ~9k vertices): capping it makes the wheel a polygon.
  """
  spec = get_spec_m22()
  spec.mesh("wheel").maxhullvert = _WHEEL_HULL_MAXVERT
  for leg in FOOT_NAMES:
    body = spec.body(f"{leg}_wheel_link")
    vis = [
      g
      for g in body.geoms
      if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == "wheel"
    ]
    assert len(vis) == 1, (leg, len(vis))
    col = spec.geom(f"{leg}_wheel_collision")
    col.type = mujoco.mjtGeom.mjGEOM_MESH
    col.meshname = "wheel"
    col.pos = vis[0].pos
    col.quat = vis[0].quat
  return spec


def unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dgv2-M22-LowNoise with the wheels colliding as the wheel.stl convex hull."""
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_env_cfg(play=play)
  cfg.scene.entities["robot"] = replace(
    cfg.scene.entities["robot"], spec_fn=get_spec_m22_hull_wheels
  )
  return cfg


def unitree_go2w_flat_aligned_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HullWheel with actuators / joint passives / wheel contact moved toward the real robot.

  See GO2W_ALIGNED_* in go2w_constants.py for the sources. On top of HullWheel:
    - legs: DcMotor (explicit PD + torque-speed curve), effort 23.7 / calf 35.55,
      armature 0.01; wheels 23.7 N·m. Command delay 0-10 ms on all joints.
    - startup randomization (per env): joint frictionloss 0-0.1 N·m (all joints),
      leg damping 0-0.05, wheel armature 0.002-0.01, wheel torsional friction
      0.005-0.02 (wheels are condim 4).
  Unchanged: kp/kd, solver (dt 0.005, iters 10/20, impratio 10), wheel friction
  0.3-1.2, mass/CoM/encoder randomization.
  """
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg(play=play)
  return apply_aligned_v1(cfg)


def apply_aligned_v1(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """The Aligned-V1 robot changes, applied in place to any Go2W velocity cfg.

  Swaps the robot's actuators, collisions (wheel condim 4) and spec (m22 with
  hull wheels; geom names unchanged, so contact sensors keep working) and adds
  the startup randomization of joint passives / wheel torsional friction.
  """
  cfg.scene.entities["robot"] = replace(
    cfg.scene.entities["robot"],
    spec_fn=get_spec_m22_hull_wheels,
    articulation=GO2W_ALIGNED_ARTICULATION,
    collisions=(ALIGNED_COLLISION,),
  )
  dr = envs_mdp.dr
  wheels = SceneEntityCfg("robot", joint_names=(".*_wheel_joint",))
  cfg.events["joint_frictionloss"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.0, 0.1)},
  )
  cfg.events["leg_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),
      "ranges": (0.0, 0.05),
    },
  )
  cfg.events["wheel_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={"asset_cfg": wheels, "ranges": (0.002, 0.01)},
  )
  cfg.events["wheel_torsional_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_GEOM_NAMES),
      "ranges": (0.005, 0.02),
      "axes": [1],
      "shared_random": True,
    },
  )
  return cfg


def apply_aligned_v2(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Aligned-V1 plus wheel rolling resistance and asymmetric wheel drag.

  - wheels condim 6, rolling friction U(0.0005, 0.003) m per wheel (independent);
  - wheel torsional friction U(0.005, 0.02) now per wheel too;
  - wheel joint damping U(0, 0.1) per wheel (v1: 0); frictionloss stays U(0, 0.1)
    per joint. Independent per-wheel drag gives left/right asymmetry, i.e. steady
    yaw torques the policy has to correct from the gyro.
  Ground friction (axis 0) stays shared by the four wheels (same floor).
  """
  cfg = apply_aligned_v1(cfg)
  cfg.scene.entities["robot"] = replace(
    cfg.scene.entities["robot"], collisions=(ALIGNED_V2_COLLISION,)
  )
  dr = envs_mdp.dr
  wheel_geoms = SceneEntityCfg("robot", geom_names=WHEEL_GEOM_NAMES)
  cfg.events["wheel_torsional_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={"asset_cfg": wheel_geoms, "ranges": (0.005, 0.02), "axes": [1]},
  )
  cfg.events["wheel_rolling_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={"asset_cfg": wheel_geoms, "ranges": (0.0005, 0.003), "axes": [2]},
  )
  cfg.events["wheel_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_wheel_joint",)),
      "ranges": (0.0, 0.1),
    },
  )
  return cfg


def unitree_go2w_flat_aligned_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HullWheel + apply_aligned_v2 (see there)."""
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_hullwheel_env_cfg(play=play)
  return apply_aligned_v2(cfg)


# Real robot weighs 22.0 kg (battery 2.5 kg + lidar/bracket, Jetson carrier,
# SSD); the model is 19.76 kg. The difference goes on base_link.
_GO2W_REAL_MASS = 22.0
_GO2W_MODEL_MASS = 19.762


def _get_spec_m22() -> mujoco.MjSpec:
  """go2w.xml with base_link made 2.24 kg heavier, inertia scaled with mass."""
  spec = get_spec()
  base = spec.body(BASE_BODY)
  s = (base.mass + _GO2W_REAL_MASS - _GO2W_MODEL_MASS) / base.mass
  base.mass *= s
  base.inertia = [x * s for x in base.inertia]
  return spec


def unitree_go2w_flat_isaacobs_yaw2_fb_low_m22_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FB-Low task at the measured real mass: 22 kg nominal, +-1 kg on the base.

  sim2sim at 22 kg reproduced the real robot's calf static error (-0.27 vs
  -0.18 rad at 19.76 kg); the policy had only ever seen 19.76 kg. The +-1 kg
  is a point mass at the base COM (dr.body_mass leaves inertia alone), on top
  of the existing base_com offset.
  """
  cfg = unitree_go2w_flat_isaacobs_yaw2_fb_low_env_cfg(play=play)
  cfg.scene.entities["robot"] = replace(
    cfg.scene.entities["robot"], spec_fn=_get_spec_m22
  )
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      "operation": "add",
      "ranges": (-1.0, 1.0),
    },
  )
  # soft_landing_peak needs every substep's force, not just the last one.
  cfg.scene.sensors = tuple(
    replace(s, history_length=cfg.decimation) if s.name == "feet_ground_contact" else s
    for s in cfg.scene.sensors
  )
  # Off by default so earlier m22 runs keep their reward set; enable on the CLI.
  # soft_landing is dropped: it missed ~1/4 of touchdowns and its reading
  # depended on the substep of touchdown (see mdp.soft_landing_peak).
  del cfg.rewards["soft_landing"]
  cfg.rewards["soft_landing_peak"] = RewardTermCfg(
    func=mdp.soft_landing_peak,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "force_threshold": 150.0,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["wheel_air_time_l2"] = RewardTermCfg(
    func=mdp.wheel_air_time_l2,
    weight=0.0,
    params={"sensor_name": "feet_ground_contact", "max_air_time": 1.0},
  )
  cfg.metrics["wheel_air_time_max"] = MetricsTermCfg(
    func=mdp.wheel_air_time_max, params={"sensor_name": "feet_ground_contact"}
  )
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

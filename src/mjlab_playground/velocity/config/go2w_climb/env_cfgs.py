"""Go2W platform climbing: 5 cm -> 50 cm ledges, two rises per tile.

New task family; nothing in config/go2w is modified, only imported.

Design (research notes in the task docstrings below and in the session log):
  - Terrain: config/go2w/terrains.py's mixed grid at 20 rows (2.25 cm a level).
    The climbing column is an inverted pyramid: spawn on the pit floor, two
    rises of h each, h = 5 -> 50 cm along the rows. mjlab's distance curriculum
    promotes at > 4 m, i.e. only after both rises (M20 / MUJICA layout).
  - Command: on the climbing column, forward only (vx 0.3-0.8) with a heading
    square to the nearest wall +-10 deg (M20's per-terrain override). Elsewhere
    upstream's omnidirectional command.
  - Rewards: planar velocity tracking in the yaw frame. mjlab's tracking terms
    add vz^2 and roll/pitch-rate^2 to the error, which pays the robot not to
    climb. upright / pose / vz penalties are switched off on the climbing column
    (a ledge above the hip needs a pitched, non-default posture); body contacts
    are a soft cost everywhere, never a termination (an early termination
    turns a scrape into a demotion: the ANYmal stair wall).
  - Robot: 22 kg model (go2w_m22.xml), +-1 kg; calf effort 35.55 N m (URDF)
    instead of 23.5, hip / thigh stay 23.5; PD 25 / 0.5 as deployed.
  - Actor: robot_lab layout (57, SDK order) + a 187-ray height scan
    (Climb), or + a climb-mode bit with 6 frames of history (Climb-Blind).
"""

import math
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.asset_zoo.robots.unitree_go2w.go2w_constants import (
  BASE_BODY,
  FOOT_NAMES,
  GO2W_WHEEL_ACTUATOR_CFG,
  get_go2w_robot_cfg,
  get_spec_m22,
)
from mjlab_playground.velocity import mdp
from mjlab_playground.velocity.config.go2w.env_cfgs import (
  _ISAAC_CLIP,
  _LEG_ACTION_SCALE,
  _LEG_JOINTS,
  _SDK_LEGS,
  _SDK_WHEELS,
  _WHEEL_ACTION_SCALE,
  _WHEEL_JOINTS,
  _make_go2w_base_cfg,
)
from mjlab_playground.velocity.config.go2w.terrains import (
  HIGH_PLATFORM_NAME,
  HIGHPLATFORM_TERRAINS_CFG,
)

from . import climb_mdp
from .climb_terrains import BoxBarRingTerrainCfg

CLIMB = HIGH_PLATFORM_NAME

##
# Terrain: 20 rows, climbing column at 40% of the envs.
##

_PROPORTIONS = {
  "flat": 0.10,
  "random_rough": 0.10,
  "hf_pyramid_slope": 0.05,
  "hf_pyramid_slope_inv": 0.05,
  "stairs_up": 0.15,
  "stairs_down": 0.15,
  CLIMB: 0.40,
}


def _climb_terrains(step_range: tuple[float, float], num_rows: int):
  subs = {
    name: replace(sub, proportion=_PROPORTIONS[name])
    for name, sub in HIGHPLATFORM_TERRAINS_CFG.sub_terrains.items()
  }
  subs[CLIMB] = replace(subs[CLIMB], step_height_range=step_range)
  return replace(HIGHPLATFORM_TERRAINS_CFG, num_rows=num_rows, sub_terrains=subs)


CLIMB_TERRAINS_CFG = _climb_terrains((0.05, 0.50), 20)
# v1 reached 100% up to 0.48 m but only 62-75% at 0.50 and ~0 above 0.53: the
# target sat on the curriculum's last row, which at saturation holds 1/20 of
# the climbing envs. Same 2.3 cm a row, carried on to 0.65 m.
CLIMB_H65_TERRAINS_CFG = _climb_terrains((0.05, 0.65), 26)

##
# Robot: 22 kg, calf at its URDF effort.
##

_HIP_THIGH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=25.0,
  damping=0.5,
  effort_limit=23.5,
)
_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=25.0,
  damping=0.5,
  effort_limit=35.55,
)


# Real-robot actuator model (the DC variants): explicit PD with a torque-speed
# curve -- full torque at rest, falling linearly to 0 at the no-load speed
# (Isaac Lab's DCMotor; robot_lab's Go2W uses 23.5 N m / 30 rad/s on every leg
# joint). The calf runs through a 1.5:1 linkage: 35.55 N m, 30 / 1.5 = 20
# rad/s. Plus a 0-15 ms command delay on every actuator (0-3 physics steps).
# MUJICA: without the curve the hind thighs ran into their limits at full
# torque while climbing and tripped the motors on the real Go2-W.
_DELAY = dict(delay_min_lag=0, delay_max_lag=3)
_DC_HIP_THIGH_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=25.0,
  damping=0.5,
  saturation_effort=23.5,
  effort_limit=23.5,
  velocity_limit=30.0,
  **_DELAY,
)
_DC_CALF_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=25.0,
  damping=0.5,
  saturation_effort=35.55,
  effort_limit=35.55,
  velocity_limit=20.0,
  **_DELAY,
)


def _climb_robot_cfg(dc: bool = False):
  if dc:
    actuators = (
      _DC_HIP_THIGH_ACTUATOR_CFG,
      _DC_CALF_ACTUATOR_CFG,
      replace(GO2W_WHEEL_ACTUATOR_CFG, **_DELAY),
    )
  else:
    actuators = (_HIP_THIGH_ACTUATOR_CFG, _CALF_ACTUATOR_CFG, GO2W_WHEEL_ACTUATOR_CFG)
  return replace(
    get_go2w_robot_cfg(),
    spec_fn=get_spec_m22,
    articulation=EntityArticulationInfoCfg(
      actuators=actuators, soft_joint_pos_limit_factor=0.9
    ),
  )


def _body_ground_sensor(name: str, geoms: tuple[str, ...]) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", entity="robot", pattern=geoms),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


_THIGH_GEOMS = tuple(f"{leg}_thigh_collision" for leg in FOOT_NAMES)
_SHANK_GEOMS = tuple(f"{leg}_calf{i}_collision" for leg in FOOT_NAMES for i in (1, 2))
_TRUNK_GEOMS = tuple(f"base{i}_collision" for i in (1, 2, 3))


def _proprio_actor_terms() -> dict[str, ObservationTermCfg]:
  """robot_lab's 57-wide actor layout in SDK order (as the IsaacObs tasks).

  Noise follows the 11-10 real-robot measurement notes (gyro +-0.05, dq +-0.5)
  instead of legged_gym's +-0.2 / +-1.5.
  """
  sdk_joints = SceneEntityCfg(
    "robot", joint_names=_SDK_LEGS + _SDK_WHEELS, preserve_order=True
  )
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
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
      noise=Unoise(n_min=-0.5, n_max=0.5),
      clip=_ISAAC_CLIP,
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, clip=_ISAAC_CLIP),
  }


def _climb_rewards() -> dict[str, RewardTermCfg]:
  legs = SceneEntityCfg("robot", joint_names=_LEG_JOINTS)
  return {
    # -- Task --------------------------------------------------------------
    "track_lin_vel": RewardTermCfg(
      func=climb_mdp.track_lin_vel_xy_yaw_exp,
      weight=2.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "track_ang_vel": RewardTermCfg(
      func=climb_mdp.track_ang_vel_z_world_exp,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # -- Posture, off the climbing column only -----------------------------
    "upright": RewardTermCfg(
      func=climb_mdp.upright_off_climb,
      weight=0.5,
      params={
        "std": math.sqrt(0.2),
        "climb_terrain_name": CLIMB,
        "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      },
    ),
    "pose": RewardTermCfg(
      func=climb_mdp.variable_posture_off_climb,
      weight=0.5,
      params={
        "asset_cfg": legs,
        "command_name": "twist",
        "climb_terrain_name": CLIMB,
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
    "lin_vel_z": RewardTermCfg(
      func=climb_mdp.lin_vel_z_l2_off_climb,
      weight=-0.5,
      params={"climb_terrain_name": CLIMB},
    ),
    # -- Regularization, everywhere ----------------------------------------
    "ang_vel_xy": RewardTermCfg(func=climb_mdp.ang_vel_xy_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": legs}
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": legs}
    ),
    "joint_torque": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    # Soft cost, 0-4 substeps per sensor per step; never a termination.
    "thigh_contact": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "thigh_ground_touch"},
    ),
    "shank_contact": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "shank_ground_touch"},
    ),
    "trunk_contact": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "trunk_ground_touch"},
    ),
    # Zero-command wheel creep: the flat tasks' sa003 (L2 on the ~0.5 s average
    # wheel speed while |command| < 0.05; final weight there -0.03). Climb-Blind
    # @2000 rolled forward at +0.82 rad/s (7 cm/s) on a zero command: the exp
    # tracking kernel charges 1.4% for that. Off by default, set on the CLI.
    # The climbing column never gets a zero command, so climbing is untouched.
    "stand_still_wheel_avg": RewardTermCfg(
      func=mdp.stand_still_wheel_vel_avg,
      weight=0.0,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS),
        "tau": 0.5,
      },
    ),
  }


def unitree_go2w_climb_env_cfg(
  play: bool = False,
  blind: bool = False,
  terrains=None,
  dc: bool = False,
  climb_bit: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Climb onto 5-50 cm ledges.

  blind=False: actor = robot_lab 57 + height scan (187), single frame.
  blind=True:  actor = robot_lab 57 + climb-mode bit, 6 frames of history; the
               bit is what an operator would switch on in front of a ledge
               (MUJICA's skill indicator / Ascento's stair-mode boolean).
  """
  cfg = _make_go2w_base_cfg()

  # -- Scene -----------------------------------------------------------------
  cfg.scene.entities = {"robot": _climb_robot_cfg(dc=dc)}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = replace(terrains or CLIMB_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 5
  # Rays from 1 m above the base, see climb_mdp.RaisedGridPatternCfg.
  cfg.scene.sensors = tuple(
    replace(
      s,
      pattern=climb_mdp.RaisedGridPatternCfg(
        size=s.pattern.size, resolution=s.pattern.resolution, z_offset=1.0
      ),
    )
    if s.name == "terrain_scan"
    else s
    for s in (cfg.scene.sensors or ())
  )
  cfg.scene.sensors = cfg.scene.sensors + (
    _body_ground_sensor("thigh_ground_touch", _THIGH_GEOMS),
    _body_ground_sensor("shank_ground_touch", _SHANK_GEOMS),
    _body_ground_sensor("trunk_ground_touch", _TRUNK_GEOMS),
  )
  # The rough Go2W task uses 500 (from upstream Go1). GPU memory scales with it:
  # at 1024 envs 9.8 GB at 500, 3.2 GB at 100, 2.3 GB at 50; 4096 envs at 500
  # took 38 of the 48 GB. MuJoCo's own default is 50; the NaN guard below
  # covers the rare EPA failure.
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 500

  # -- Actions: robot_lab / SDK order ------------------------------------------
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

  # -- Observations --------------------------------------------------------------
  actor = cfg.observations["actor"]
  actor.terms = _proprio_actor_terms()
  if blind:
    if climb_bit:
      actor.terms["climb_mode"] = ObservationTermCfg(
        func=climb_mdp.climb_indicator, params={"terrain_name": CLIMB}
      )
    actor.history_length = 6
    actor.flatten_history_dim = True
  else:
    actor.terms["height_scan"] = ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=1 / 5.0,  # terrain_scan.max_distance
    )
  critic = cfg.observations["critic"].terms
  critic["climb_mode"] = ObservationTermCfg(
    func=climb_mdp.climb_indicator, params={"terrain_name": CLIMB}
  )
  for name in ("thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch"):
    critic[name] = ObservationTermCfg(
      func=climb_mdp.contact_found, params={"sensor_name": name}
    )

  # -- Commands --------------------------------------------------------------------
  cfg.commands["twist"] = climb_mdp.ClimbAwareVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    rel_forward_envs=0.2,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
    climb_terrain_name=CLIMB,
    climb_lin_vel_x=(0.3, 0.8),
    climb_heading_noise=math.radians(10.0),
  )

  # -- Rewards / metrics -------------------------------------------------------------
  cfg.rewards = _climb_rewards()
  cfg.metrics["track_lin_vel_err"] = MetricsTermCfg(
    func=mdp.track_lin_vel_err, params={"command_name": "twist"}
  )
  cfg.metrics["track_yaw_rate_err"] = MetricsTermCfg(
    func=mdp.track_yaw_rate_err, params={"command_name": "twist"}
  )

  # -- Terminations: only falling over (80 deg; a ledge climb pitches ~40-50) ----
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(80.0)}
    ),
    # Upstream truncates 0.3 m *inside* the grid edge, i.e. 3.7 m from the
    # centre of an edge tile: a robot climbing out towards the outside of the
    # climbing column (the last one) never reaches the 4 m promotion distance
    # and is demoted for its success. The 20 m border is flat at z = 0, so
    # only truncate 10 m out.
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds, time_out=True, params={"margin": -10.0}
    ),
    # Safety net for an unattended 10 h run: a NaN world (wheel vs box EPA,
    # mujoco_warp#1782) is reset instead of killing the run. Rewards are
    # already nan_to_num'd by the reward manager; obs are sanitized below.
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }
  for group in ("actor", "critic"):
    cfg.observations[group].nan_policy = "sanitize"

  # -- Curriculum: terrain only, commands fixed --------------------------------------
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel, params={"command_name": "twist"}
    ),
  }

  # -- Events ----------------------------------------------------------------------
  cfg.events["push_robot"] = replace(
    cfg.events["push_robot"],
    interval_range_s=(8.0, 12.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
  )
  cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      "operation": "add",
      "ranges": (-1.0, 1.0),
    },
  )

  if dc:
    # kp / kd x U(0.85, 1.15) per joint, redrawn every episode (MUJICA: motor
    # gain multiplier 0.8-1.2; M20: 0.85-1.15).
    cfg.events["pd_gains"] = EventTermCfg(
      mode="reset",
      func=climb_mdp.scale_pd_gains,
      params={"scale_range": (0.85, 1.15)},
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    # Curriculum layout kept (column = terrain type), every row populated.
    cfg.scene.terrain.max_init_terrain_level = None

  return cfg


def unitree_go2w_climb_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return unitree_go2w_climb_env_cfg(play=play, blind=True)


def unitree_go2w_climb_h65_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Climb with the ledge range carried on to 0.65 m (26 rows), nothing else."""
  return unitree_go2w_climb_env_cfg(play=play, terrains=CLIMB_H65_TERRAINS_CFG)


def unitree_go2w_climb_blind_h65_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return unitree_go2w_climb_env_cfg(
    play=play, blind=True, terrains=CLIMB_H65_TERRAINS_CFG
  )


def unitree_go2w_climb_blind_dc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Climb-Blind with the real-robot actuator model: DC torque-speed curve,
  0-15 ms command delay, kp / kd +-15% per episode. Same observations and
  actions, so Climb-Blind checkpoints load as-is."""
  return unitree_go2w_climb_env_cfg(play=play, blind=True, dc=True)


def unitree_go2w_climb_blind_dc_h65_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Climb-Blind-DC with the ledge range carried on to 0.65 m (26 rows)."""
  return unitree_go2w_climb_env_cfg(
    play=play, blind=True, dc=True, terrains=CLIMB_H65_TERRAINS_CFG
  )


# Columns where the flat tasks' gait terms apply (the stairs and the climb are
# left to themselves: a trot score and a 4 cm swing target are not what a step
# or a ledge needs).
_POSTURE_TERRAINS = ("flat", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv")


def unitree_go2w_climb_blind_dc_h65_flatpose_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Climb-Blind-DC-H65, posture brought towards the flat task, trunk off the ledge.

  Bench of the Climb policies (tools/climb_bench.py, posture_bench.py) found:
    - 33-83% of the climbs put the trunk on the ledge, p95 peak ~800-1200 N
      (soft cost -0.1 per substep: ~0.4 a step against 2.0 for tracking).
      MUJICA terminates on base collision and constrains thigh/calf contacts.
    - off the ledge, slow turns ran on a twisted stance (|dq| hip/calf 0.23 rad
      at wz 0.15, flat dgv2 0.03 / 0.09) and stepping was a 4 Hz shuffle (air
      5-24%) instead of the flat task's 2 Hz trot (air 44%).
  Changes, all else as the parent:
    - command: the flat tasks' buckets (standing / vx / vy / wz / slow / combo,
      deadzone 0.05, wz +-2) off the climbing column; climbing column as before.
    - tracking: on the climbing column unchanged (x2 inside the term); elsewhere
      the flat tasks' cmd-scaled terms at weight 1, which carry vz and the
      roll / pitch rates, so lin_vel_z is dropped.
    - dgv2 0.5 and foot_swing_clearance -0.35 (a8ea1db's settings) on the flat,
      rough and slope columns only.
    - stand_still_wheel_avg -0.03 (sa003) on.
    - trunk contact -1.0 (was -0.1), thigh -0.3 (was -0.1), shank unchanged:
      hooking the shanks over the edge is how ANYmal and MUJICA climb.
  The critic has no dgv2 window summary, so parent checkpoints load as-is.
  """
  cfg = unitree_go2w_climb_env_cfg(
    play=play, blind=True, dc=True, terrains=CLIMB_H65_TERRAINS_CFG
  )
  _apply_flatpose(cfg)
  return cfg


def _apply_flatpose(cfg: ManagerBasedRlEnvCfg) -> None:
  """FlatPose's command and reward changes (see that task), applied in place."""
  old = cfg.commands["twist"]
  cfg.commands["twist"] = climb_mdp.ClimbAwareBucketVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=old.resampling_time_range,
    heading_command=False,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-2.0, 2.0), heading=None
    ),
    deadzone=0.05,
    climb_terrain_name=CLIMB,
    climb_lin_vel_x=(0.3, 0.8),
    climb_heading_noise=math.radians(10.0),
  )
  r = cfg.rewards
  r["track_lin_vel"] = RewardTermCfg(
    func=climb_mdp.track_lin_vel_climb_or_flat,
    weight=1.0,
    params={"command_name": "twist", "std": 0.5, "climb_terrain_name": CLIMB},
  )
  r["track_ang_vel"] = RewardTermCfg(
    func=climb_mdp.track_ang_vel_climb_or_flat,
    weight=1.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.5),
      "climb_terrain_name": CLIMB,
    },
  )
  r["lin_vel_z"].weight = 0.0
  r["dgv2"] = RewardTermCfg(
    func=climb_mdp.diagonal_gait_v2_on,
    weight=0.5,
    params={
      "posture_terrains": _POSTURE_TERRAINS,
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "period": 0.5,
      "sigma": 0.05,
      "contact_threshold": 0.8,
      "full_load_frac": 0.5,
      "straight_threshold": 1e-3,
      "balance_power": 2.0,
      "require_all_down": True,
    },
  )
  r["foot_swing_clearance"] = RewardTermCfg(
    func=climb_mdp.feet_swing_clearance_on,
    weight=-0.35,
    params={
      "posture_terrains": _POSTURE_TERRAINS,
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_clearance": 0.04,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  r["stand_still_wheel_avg"].weight = -0.03
  r["trunk_contact"].weight = -1.0
  r["thigh_contact"].weight = -0.3


BAR = "bar"
_BAR_PROPORTIONS = {
  "flat": 0.10,
  "random_rough": 0.10,
  "hf_pyramid_slope": 0.05,
  "hf_pyramid_slope_inv": 0.05,
  "stairs_up": 0.10,
  "stairs_down": 0.10,
  CLIMB: 0.25,
  BAR: 0.25,
}
CLIMB_BAR_TERRAINS_CFG = replace(
  CLIMB_H65_TERRAINS_CFG,
  num_cols=len(_BAR_PROPORTIONS),
  sub_terrains={
    **{
      name: replace(sub, proportion=_BAR_PROPORTIONS[name])
      for name, sub in CLIMB_H65_TERRAINS_CFG.sub_terrains.items()
    },
    # The 40 x 40 cm long-bar test, as a climbing column: same 26 rows over
    # 5-65 cm as the ledges, bar 30-60 cm wide.
    BAR: BoxBarRingTerrainCfg(
      proportion=_BAR_PROPORTIONS[BAR],
      height_range=(0.05, 0.65),
      bar_width_range=(0.3, 0.6),
      ring_distance=2.0,
    ),
  },
)


def _retarget_climb(cfg: ManagerBasedRlEnvCfg, names: str) -> None:
  """Point every climb-column parameter (command, rewards, observations) at
  ``names`` ("a|b"), i.e. treat all those columns as climbing columns."""
  for key in ("climb_terrain_name", "terrain_name"):
    for group in (cfg.rewards, *(g.terms for g in cfg.observations.values())):
      for term in group.values():
        if term is not None and term.params.get(key) == CLIMB:
          term.params[key] = names
  cmd = cfg.commands["twist"]
  if getattr(cmd, "climb_terrain_name", None) == CLIMB:
    cmd.climb_terrain_name = names


def unitree_go2w_climb_blind_dc_h65_flatpose_bar_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FlatPose plus the long-bar column (climb_terrains.BoxBarRingTerrainCfg).

  Target 11-10: get over a 40 cm high, 40 cm wide long bar without falling.
  Climb-Blind-DC-H65@2750 crossed it zero-shot at 30-50 cm (100%, no falls) by
  straddling it on its belly. The bar is a climbing column: climb bit on,
  forward-only command square to the bar, no posture terms. A 40 cm bar is
  shorter than the body, so belly contact is sometimes unavoidable: the trunk
  stays a cost (-1.0), not a termination. Observations unchanged.
  """
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CLIMB_BAR_TERRAINS_CFG)
  _retarget_climb(cfg, f"{CLIMB}|{BAR}")
  return cfg


# Bar widths 15-80 cm: FlatPose-Bar@4000 (trained on 30-60 cm) crossed 20 cm wide
# bars only 81-88% at 40-50 cm (Blind-DC-H65@3000: 25-81%), stuck high-centred
# on the belly, wheels off the ground.
CLIMB_BAR2_TERRAINS_CFG = replace(
  CLIMB_BAR_TERRAINS_CFG,
  sub_terrains={
    **CLIMB_BAR_TERRAINS_CFG.sub_terrains,
    BAR: replace(CLIMB_BAR_TERRAINS_CFG.sub_terrains[BAR], bar_width_range=(0.15, 0.8)),
  },
)


def unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FlatPose-Bar with bar widths 15-80 cm instead of 30-60. Observations unchanged."""
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_bar_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CLIMB_BAR2_TERRAINS_CFG)
  return cfg


def unitree_go2w_climb_blind_dc_h65_flatpose_bar2_angle45_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FlatPose-Bar2 with the climbing-column heading noise widened from +-10 to
  +-45 deg, so the robot meets ledges and bars at up to 45 deg off square.

  v6b (Bar2)@8500: 100% up at 0/15 deg, but at 30 deg the >=80% height fell to
  0.49 m (0.55 m+ fell 31-62%), at 40 deg to 0.40 m. Observations unchanged.
  """
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg(play=play)
  cfg.commands["twist"].climb_heading_noise = math.radians(45.0)
  return cfg


def unitree_go2w_climb_blind_dc_h65_flatpose_bar2_angle30_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Angle45's control: heading noise +-30 deg."""
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg(play=play)
  cfg.commands["twist"].climb_heading_noise = math.radians(30.0)
  return cfg


def unitree_go2w_climb_blind_dc_h65_flatpose_bar2_stop_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FlatPose-Bar2 with stops on the climbing columns and wheel friction 0.3-1.0.

  FlatPose-Bar2@5500: zero command with climb mode on fell 1-2% (on top of a
  ledge, or halfway up) -- the climbing columns never got a zero command; and
  at wheel friction 0.3 (training 0.5-1.0) ledges of 45 cm and up dropped to
  4-25% with 13% falls. Now 15% of the climbing-column resamples are a stop,
  and the wheel friction is drawn from 0.3-1.0. Observations unchanged.
  """
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_bar2_env_cfg(play=play)
  cfg.commands["twist"].climb_stop_prob = 0.15
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.0)
  return cfg


def unitree_go2w_climb_dc_h65_flatpose_bar_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Height-scan sibling of FlatPose-Bar: actor = robot_lab 57 + the 187-ray scan,
  single frame, DC motors; everything else as FlatPose-Bar.

  Climb (scan) 33ab333@1000 crossed the 40 cm bar 100% with no trunk contact at
  all (blind policies: 62-100% of the crossings on the belly); seeing the bar,
  it steps over instead of running into it. Same observations as the Climb
  task, so its checkpoints load as-is. Needs an elevation map on the robot.
  """
  cfg = unitree_go2w_climb_env_cfg(play=play, dc=True, terrains=CLIMB_H65_TERRAINS_CFG)
  _apply_flatpose(cfg)
  cfg.scene.terrain.terrain_generator = replace(CLIMB_BAR_TERRAINS_CFG)
  _retarget_climb(cfg, f"{CLIMB}|{BAR}")
  return cfg


def unitree_go2w_climb_dc_h65_flatpose_bar2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Scan FlatPose-Bar with bar widths 15-80 cm: @2000 (trained on 30-60 cm)
  crossed 20 cm wide bars only 12-31% at 40-50 cm, stuck (the blind sibling 81-88%).
  Observations unchanged."""
  cfg = unitree_go2w_climb_dc_h65_flatpose_bar_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CLIMB_BAR2_TERRAINS_CFG)
  return cfg


def unitree_go2w_climb_blind_dc_h65_flatpose_trunkterm_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FlatPose with the trunk on the terrain as a termination (MUJICA: an episode
  ends on base collision), for when the -1.0 cost does not get the trunk off
  the ledge. A terminated climber travels less and is demoted, so the
  curriculum settles where the policy can climb without its belly.
  Observations unchanged: FlatPose / Blind-DC-H65 checkpoints load as-is."""
  cfg = unitree_go2w_climb_blind_dc_h65_flatpose_env_cfg(play=play)
  if not play:
    cfg.terminations["trunk_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "trunk_ground_touch", "force_threshold": 10.0},
    )
  return cfg


def unitree_go2w_climb_blind_nobit_dc_h65_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Climb-Blind-DC-H65 without the climb-mode bit: robot_lab 57 x 6 frames.

  Off the climbing column there is no wall the robot should *not* climb, so
  "bumped into a ledge -> climb" never conflicts with anything else; if this
  climbs as well as the bit version, the deploy side needs no mode switch.
  """
  return unitree_go2w_climb_env_cfg(
    play=play, blind=True, dc=True, terrains=CLIMB_H65_TERRAINS_CFG, climb_bit=False
  )

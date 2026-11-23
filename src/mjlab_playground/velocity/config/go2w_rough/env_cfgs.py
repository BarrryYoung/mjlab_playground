"""Go2W rough terrain: the flat dgv2 policy carried onto stairs, slopes and rough ground.

New task family; nothing in config/go2w or config/go2w_climb is modified, only
imported.

Built on the flat Cmd-Dgv2-M22 task, warm-started from a8ea1db@7999:
  - Rewards: that run's CLI settings (tl1 ta1 p05 a001 fsc035-004 u05 sa003
    dgv2w05 bp2 alldown push10) as defaults. dgv2 and the swing clearance are
    only paid on the flat / rough / slope columns (a trot score and a 4 cm swing
    target are not what a stair needs); upright is measured against the terrain
    normal; thigh / shank / trunk ground contacts are a soft cost (as v6b).
  - Terrain: the climb grid without the high platform and the bar, stairs
    2-18 cm (30 cm treads) over 10 rows.
  - Robot: v6b's DC motor model (torque-speed curve, 0-15 ms delay, kp / kd
    +-15%), 22 kg +-1 kg, wheel friction 0.3-1.0.
  - Actor: unchanged (robot_lab 57, single frame), so the policy loads as-is.
  - Critic: the flat critic (85) + the 187-ray height scan appended last;
    tools/warmstart_ckpt.py pads the first critic layer with zero columns.
"""

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.velocity import mdp
from mjlab_playground.velocity.config.go2w.env_cfgs import (
  apply_aligned_v1,
  apply_aligned_v2,
  _make_go2w_base_cfg,
  unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg,
)
from mjlab_playground.velocity.config.go2w.terrains import (
  HIGH_PLATFORM_NAME,
  HIGHPLATFORM_TERRAINS_CFG,
)
from mjlab_playground.velocity.config.go2w_climb import climb_mdp
from mjlab_playground.velocity.config.go2w_climb.env_cfgs import (
  _POSTURE_TERRAINS,
  _SHANK_GEOMS,
  _THIGH_GEOMS,
  _TRUNK_GEOMS,
  _body_ground_sensor,
  _climb_robot_cfg,
)
from mjlab_playground.velocity.config.go2w_rough import rough_mdp

##
# Terrain: the climb grid minus the climbing columns.
##

_PROPORTIONS = {
  "flat": 0.10,
  "random_rough": 0.10,
  "hf_pyramid_slope": 0.05,
  "hf_pyramid_slope_inv": 0.05,
  "stairs_up": 0.10,
  "stairs_down": 0.10,
}
STAIR_STEP_RANGE = (0.02, 0.18)

ROUGH_TERRAINS_CFG = replace(
  HIGHPLATFORM_TERRAINS_CFG,
  num_rows=10,
  sub_terrains={
    name: replace(sub, proportion=_PROPORTIONS[name])
    for name, sub in HIGHPLATFORM_TERRAINS_CFG.sub_terrains.items()
    if name != HIGH_PLATFORM_NAME
  },
)
for _name in ("stairs_up", "stairs_down"):
  ROUGH_TERRAINS_CFG.sub_terrains[_name] = replace(
    ROUGH_TERRAINS_CFG.sub_terrains[_name], step_height_range=STAIR_STEP_RANGE
  )


def _terrain_scan():
  """The base cfg's terrain_scan, rays raised 1 m (an 18 cm stair pyramid puts
  steps above the base within the 1.6 x 1.0 m grid; see RaisedGridPatternCfg)."""
  scan = next(
    s for s in _make_go2w_base_cfg().scene.sensors if s.name == "terrain_scan"
  )
  return replace(
    scan,
    pattern=climb_mdp.RaisedGridPatternCfg(
      size=scan.pattern.size, resolution=scan.pattern.resolution, z_offset=1.0
    ),
  )


def unitree_go2w_rough_dgv2_m22_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2w_flat_isaacobs_cmd_dgv2_m22_env_cfg(play=play)

  # -- Scene -----------------------------------------------------------------
  cfg.scene.entities = {"robot": _climb_robot_cfg(dc=True)}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = replace(ROUGH_TERRAINS_CFG, curriculum=True)
  # The flat policy has never seen a step: start on the two easiest rows.
  cfg.scene.terrain.max_init_terrain_level = 1
  scan = _terrain_scan()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    scan,
    _body_ground_sensor("thigh_ground_touch", _THIGH_GEOMS),
    _body_ground_sensor("shank_ground_touch", _SHANK_GEOMS),
    _body_ground_sensor("trunk_ground_touch", _TRUNK_GEOMS),
  )
  # The flat task runs njmax 300 / ccd 50 / maxmatch 64; the climb settings.
  cfg.sim.njmax = 1500
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 500

  # -- Observations: critic gets the scan, appended last ----------------------
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / scan.max_distance,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].nan_policy = "sanitize"

  # -- Rewards: a8ea1db's CLI settings ----------------------------------------
  r = cfg.rewards
  for name in ("track_linear_velocity", "track_angular_velocity"):
    r[name].weight = 1.0
    r[name].params.update(cmd_scaled=True, std_min=0.1)
  r["upright"].weight = 0.5
  r["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  r["pose"].weight = 0.5
  r["dof_pos_limits"].weight = 0.0
  r["action_rate_l2"].weight = -0.01
  r["foot_clearance"].weight = 0.0
  r["foot_slip"].weight = 0.0
  r["soft_landing"].weight = 0.0
  r["stand_still_wheel_avg"].weight = -0.03
  r["stand_still_wheel_avg"].params["max_vel"] = 3.0
  r["dgv2"] = replace(
    r["dgv2"],
    func=climb_mdp.diagonal_gait_v2_on,
    weight=0.5,
    params={
      **r["dgv2"].params,
      "balance_power": 2.0,
      "require_all_down": True,
      "posture_terrains": _POSTURE_TERRAINS,
      # A heading controller writes wz every step and settles at a degree or two
      # (0.001-0.05): charge neither trot nor balance there.
      "steer_deadband": 0.05,
    },
  )
  r["foot_swing_clearance"] = replace(
    r["foot_swing_clearance"],
    func=climb_mdp.feet_swing_clearance_on,
    weight=-0.35,
    params={**r["foot_swing_clearance"].params, "posture_terrains": _POSTURE_TERRAINS},
  )
  # v6b's contact costs: 0-4 substeps per sensor per step, never a termination.
  for name, weight in (("thigh", -0.3), ("shank", -0.1), ("trunk", -1.0)):
    r[f"{name}_contact"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=weight,
      params={"sensor_name": f"{name}_ground_touch"},
    )

  # -- Terminations ----------------------------------------------------------
  # Upstream truncates 0.3 m inside the grid edge, 3.7 m from the centre of an
  # edge tile: robots heading out of an edge column never reach the 4 m
  # promotion distance. The 20 m border is flat, so only truncate 10 m out.
  cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=mdp.out_of_terrain_bounds, time_out=True, params={"margin": -10.0}
  )
  # Wheel vs stair-box EPA can still NaN a world (mujoco_warp#1782): reset it.
  cfg.terminations["nan"] = TerminationTermCfg(func=envs_mdp.nan_detection)

  # -- Curriculum: terrain only, commands fixed --------------------------------
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel, params={"command_name": "twist"}
    ),
  }

  # -- Events ----------------------------------------------------------------
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (9.0, 11.0)
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.0)
  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset",
    func=climb_mdp.scale_pd_gains,
    params={"scale_range": (0.85, 1.15)},
  )

  if play:
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    # Curriculum layout kept (column = terrain type), every row populated.
    cfg.scene.terrain.max_init_terrain_level = None

  return cfg


def unitree_go2w_rough_dgv2_m22_track_l0_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Rough-Dgv2-M22 pinned to the easiest row, scored on velocity tracking.

  - Command: the bucket command on flat / rough / slopes; on the stairs straight
    forward / backward driving with the heading held from spawn for the episode.
  - Curriculum: no promotion. tracking_curriculum only logs, per column, the
    per-episode tracking score, the share of episodes that would be promoted
    (no fall, score >= 0.6) and the fall rate. This is the baseline before
    switching promotion on with the same criterion.
  """
  cfg = unitree_go2w_rough_dgv2_m22_env_cfg(play=play)
  old = cfg.commands["twist"]
  cfg.commands["twist"] = rough_mdp.TerrainAwareBucketVelocityCommandCfg(
    **{f.name: getattr(old, f.name) for f in fields(old)}
  )
  if not play:
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = {
      "tracking": CurriculumTermCfg(
        func=rough_mdp.tracking_curriculum,
        params={"command_name": "twist", "promote": False},
      ),
    }
  return cfg


def unitree_go2w_rough_dgv2_m22_track_l0_wz_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """TrackL0 with the angular tracking reward split by terrain: on flat it
  scores the yaw rate and the roll / pitch rates (as before); everywhere else
  only the yaw rate. Observations unchanged, so TrackL0 checkpoints load."""
  cfg = unitree_go2w_rough_dgv2_m22_track_l0_env_cfg(play=play)
  term = cfg.rewards["track_angular_velocity"]
  cfg.rewards["track_angular_velocity"] = replace(
    term,
    func=rough_mdp.track_angular_velocity_terrain,
    params={**term.params, "full_terrains": ("flat",)},
  )
  return cfg


def unitree_go2w_rough_dgv2_m22_track_l0_wz_h6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """TrackL0-Wz with 6 frames of actor history (57 x 6 = 342), as the Climb-Blind
  policies. The blind single-frame actor cannot tell a step from flat ground;
  the history shows the wheels being stopped and the base lifting. Warm-start
  with tools/warmstart_ckpt.py: the old weights go on the newest frame, the
  five older frames start at zero."""
  cfg = unitree_go2w_rough_dgv2_m22_track_l0_wz_env_cfg(play=play)
  actor = cfg.observations["actor"]
  actor.history_length = 6
  actor.flatten_history_dim = True
  return cfg


def unitree_go2w_rough_heading_h6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """TrackL0-Wz-H6 with an ANYmal-rough command and a terrain curriculum.

  - Command: the bucket command on flat; on rough / slopes / stairs vx U(-1, 1),
    vy 0, a random world heading held by the P controller (wz <= 1), 2%
    standing, 10 s per command.
  - Curriculum: from row 0. Off flat, up a row on net distance > 4 m and
    tracking score >= 0.6; down a row on distance < half the commanded one.
    Flat envs stay put. tracking_curriculum still logs the per-column scores.
  - dgv2 0.2 (warm-start from the dgv2w02 flat run d9d5fa1@7999).
  """
  cfg = unitree_go2w_rough_dgv2_m22_track_l0_wz_h6_env_cfg(play=play)
  # Vertical speed is part of the tracking error only on flat ground: climbing
  # a stair means the base moves up, and charging that costs the term there.
  lin = cfg.rewards["track_linear_velocity"]
  cfg.rewards["track_linear_velocity"] = replace(
    lin,
    func=rough_mdp.track_linear_velocity_terrain,
    params={**lin.params, "full_terrains": ("flat",)},
  )
  old = cfg.commands["twist"]
  cfg.commands["twist"] = rough_mdp.HeadingRoughVelocityCommandCfg(
    **{
      f.name: getattr(old, f.name) for f in fields(old) if f.name != "straight_terrains"
    }
  )
  cfg.rewards["dgv2"].weight = 0.2
  if not play:
    cfg.curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=rough_mdp.terrain_levels_vel_on, params={"command_name": "twist"}
      ),
      "tracking": CurriculumTermCfg(
        func=rough_mdp.tracking_curriculum,
        params={"command_name": "twist", "promote": False},
      ),
    }
  return cfg


def unitree_go2w_rough_heading_fwd_h6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Heading-H6 with the 11-17 design:

  - Command: flat and random_rough use the flat bucket command; stairs and slopes
    a forward-only heading command (vx U(0.3, 1), vy 0, a world heading
    U(-pi, pi) held for the whole episode), no standing. Backward is left out: the calves point
    back, a reversing robot scrapes them on the steps and would learn to stretch
    them out behind.
  - Tracking: random_rough is scored like flat (vz, roll / pitch rate, command-
    scaled std); stairs and slopes keep the relaxed version.
  - dgv2 (weight 0.5, as the a8ea1db warm start) and the swing clearance: on flat, random_rough and the slopes, and
    anywhere the scan under the base is flat (std < 1.5 cm); off on stair tiles.
  - Curriculum: random_rough does not promote (its levels are all alike and its
    bucket command does not travel).
  """
  cfg = unitree_go2w_rough_heading_h6_env_cfg(play=play)
  full = ("flat", "random_rough")
  for name in ("track_linear_velocity", "track_angular_velocity"):
    term = cfg.rewards[name]
    cfg.rewards[name] = replace(term, params={**term.params, "full_terrains": full})
  old = cfg.commands["twist"]
  cfg.commands["twist"] = replace(
    old,
    bucket_terrains=full,
    heading_lin_vel_x=(0.3, 1.0),
    heading_standing_prob=0.0,
    heading_resampling_time=1e6,  # one heading per episode
  )
  gate = {"scan_sensor_name": "terrain_scan", "flat_std": 0.015}
  dgv2 = cfg.rewards["dgv2"]
  cfg.rewards["dgv2"] = replace(
    dgv2,
    func=rough_mdp.diagonal_gait_v2_gated,
    weight=0.5,  # warm-start from the flat dgv2 w05 run (a8ea1db), not w02
    params={**dgv2.params, **gate},
  )
  fsc = cfg.rewards["foot_swing_clearance"]
  cfg.rewards["foot_swing_clearance"] = replace(
    fsc, func=rough_mdp.feet_swing_clearance_gated, params={**fsc.params, **gate}
  )
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=rough_mdp.terrain_levels_vel_on,
      params={"command_name": "twist", "skip_terrains": full},
    )
  return cfg


_STRAIGHT_TERRAINS = (
  "hf_pyramid_slope",
  "hf_pyramid_slope_inv",
  "stairs_up",
  "stairs_down",
)


def unitree_go2w_rough_straight_h6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Heading-Fwd-H6 without the heading controller (11-18).

  The heading controller wrote wz every step on slopes and stairs and hid a
  steady yaw drift of the policy (about -0.09 rad/s, also on flat ground: the
  policy cannot see the terrain). Here slopes and stairs get vx U(0.3, 1) only,
  wz 0, no standing, from a random spawn yaw (the reset event draws +-pi), and a
  bounded drift penalty on the heading change since the spawn:
  -0.3 * min(|yaw - yaw_spawn| / 0.8 rad, 1). The critic sees that error.
  """
  cfg = unitree_go2w_rough_heading_fwd_h6_env_cfg(play=play)
  old = cfg.commands["twist"]
  base_fields = {f.name for f in fields(rough_mdp.TerrainAwareBucketVelocityCommandCfg)}
  cfg.commands["twist"] = rough_mdp.TerrainAwareBucketVelocityCommandCfg(
    **{
      **{f.name: getattr(old, f.name) for f in fields(old) if f.name in base_fields},
      "straight_terrains": _STRAIGHT_TERRAINS,
      "straight_lin_vel_x": (0.3, 1.0),
      "straight_backward_prob": 0.0,
      "straight_hold_heading": False,
    }
  )
  cfg.rewards["heading_drift"] = RewardTermCfg(
    func=rough_mdp.heading_drift_penalty,
    weight=-0.3,
    params={"terrains": _STRAIGHT_TERRAINS, "tolerance": 0.8},
  )
  # 腿部力矩平方和惩罚(不含轮子);unitree_rl_lab Go2 / LocoWheeledLegged 都是 -2e-4。
  cfg.rewards["joint_torques_l2"].weight = -2e-4
  cfg.observations["critic"].terms["heading_err"] = ObservationTermCfg(
    func=rough_mdp.heading_error_from_spawn,
    params={"terrains": _STRAIGHT_TERRAINS},
  )
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Straight-H6 on the flat lownoise_ht0003 recipe (warm start: lownoise_ht0003@7999).

  Same two changes the flat run made on top of w05: actor noise ang_vel +-0.05 and
  joint_vel +-0.5 (measured on the real robot), and hip_torques_l2 -0.003 (hips only,
  straight / standing only). The leg-wide joint_torques_l2 stays at 0 as in the flat run.
  """
  cfg = unitree_go2w_rough_straight_h6_env_cfg(play=play)
  terms = cfg.observations["actor"].terms
  for name, a in (("base_ang_vel", 0.05), ("joint_vel", 0.5)):
    assert isinstance(terms[name].noise, Unoise), name
    terms[name] = replace(terms[name], noise=Unoise(n_min=-a, n_max=a))
  cfg.rewards["hip_torques_l2"].weight = -0.003
  cfg.rewards["joint_torques_l2"].weight = 0.0
  return cfg


def unitree_go2w_rough_straight_h6_lownoise_ht0003_aligned_v1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Straight-H6-LowNoise-Ht0003 with the flat Aligned-V1 robot (apply_aligned_v1).

  Replaces the climb DC actuators (sat = effort, curve from 0 rad/s; delay 0-15 ms)
  with the Aligned-V1 ones (Go2HV knee at 13.5 / calf 9 rad/s, effort 23.7 / 35.55,
  armature 0.01, delay 0-10 ms sticky), switches the wheels to the hull collider
  with condim 4, and randomizes joint frictionloss / leg damping / wheel armature /
  wheel torsional friction. Rewards, commands, terrain, obs unchanged.
  """
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play)
  return apply_aligned_v1(cfg)


def unitree_go2w_rough_straight_h6_lownoise_ht0003_aligned_v2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Straight-H6-LowNoise-Ht0003 with the Aligned-V2 robot (apply_aligned_v2):
  Aligned-V1 plus wheel rolling friction (condim 6) and per-wheel drag."""
  cfg = unitree_go2w_rough_straight_h6_lownoise_ht0003_env_cfg(play=play)
  return apply_aligned_v2(cfg)

"""Unitree Go2W constants.

MJCF: xmls/go2w.xml -- Unitree's official mjlab go2.xml with wheels added; see the
header comment there for provenance. Link masses follow the go2w_description URDF
that robot_lab trains on.

Actuator gains, limits and init pose mirror robot_lab's UNITREE_GO2W_CFG.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
  DcMotorActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2W_XML: Path = Path(__file__).parent / "xmls" / "go2w.xml"
assert GO2W_XML.exists()
# Same model at the real robot's 22.0 kg (heavier base_link only).
GO2W_M22_XML: Path = Path(__file__).parent / "xmls" / "go2w_m22.xml"
assert GO2W_M22_XML.exists()

FOOT_NAMES = ("FR", "FL", "RR", "RL")
WHEEL_SITE_NAMES = tuple(f"{leg}_wheel" for leg in FOOT_NAMES)
WHEEL_GEOM_NAMES = tuple(f"{leg}_wheel_collision" for leg in FOOT_NAMES)
BASE_BODY = "base_link"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2W_XML))


def get_spec_m22() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2W_M22_XML))


##
# Actuator config (robot_lab: legs DCMotor kp=25 kd=0.5 23.5 N·m; wheels
# velocity-only kd=0.5 23.5 N·m).
##

GO2W_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
  stiffness=25.0,
  damping=0.5,
  effort_limit=23.5,
)
GO2W_WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  damping=0.5,
  effort_limit=23.5,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.45),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_wheel_regex = "^[FR][LR]_wheel_collision$"

# All colliders on, self collisions off (conaffinity=0), as in Unitree's go2 and
# robot_lab's enabled_self_collisions=False. Non-wheel colliders are frictionless.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_wheel_regex: 3, ".*_collision": 1},
  priority={_wheel_regex: 1},
  friction={_wheel_regex: (0.6,)},
  contype=1,
  conaffinity=0,
)

##
# "Aligned" config (2026-11-19, task *-Aligned-V1): actuator limits and passive
# joint terms moved toward the real robot. Sources:
#   - effort / velocity: unitree_ros go2w_description URDF (hip/thigh/wheel 23.7 N·m
#     30.1 rad/s; calf 35.55 N·m 20.07 rad/s = x1.5 linkage reduction).
#   - torque-speed curve: unitree_rl_lab UnitreeActuatorCfg_Go2HV (full torque up to
#     13.5 rad/s, linear to 0 at 30 rad/s); calf scaled by the 1.5 reduction (9 / 20).
#     DcMotor: limit = sat * (1 - v / vmax) capped at effort_limit, so
#     sat = effort / (1 - knee / vmax).
#   - armature 0.01: unitree_rl_mjlab Go2 hip/thigh and unitree_mujoco go2w.xml;
#     Go1 rotor 1.12e-4 * (6.33 * 1.5)^2 gives ~0.010 for the Go2-W calf too.
#   - command delay 0-2 physics steps (0-10 ms), sticky (hold prob 0.8): guess, not
#     measured on the real robot.
# Wheel armature, joint friction/damping and wheel torsional friction are
# randomized per env in the task (env_cfgs.py), not set here.
##

_ALIGNED_DELAY = dict(delay_min_lag=0, delay_max_lag=2, delay_hold_prob=0.8)


def _dc_sat(effort: float, knee: float, vmax: float) -> float:
  return effort / (1.0 - knee / vmax)


GO2W_ALIGNED_HIP_THIGH_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=25.0,
  damping=0.5,
  effort_limit=23.7,
  saturation_effort=_dc_sat(23.7, 13.5, 30.1),
  velocity_limit=30.1,
  armature=0.01,
  **_ALIGNED_DELAY,
)
GO2W_ALIGNED_CALF_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=25.0,
  damping=0.5,
  effort_limit=35.55,
  saturation_effort=_dc_sat(35.55, 9.0, 20.07),
  velocity_limit=20.07,
  armature=0.01,
  **_ALIGNED_DELAY,
)
# Wheels stay a builtin velocity actuator (no torque-speed curve; ~16 rad/s at our
# command range, under the 30 rad/s URDF limit).
GO2W_ALIGNED_WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  damping=0.5,
  effort_limit=23.7,
  **_ALIGNED_DELAY,
)

GO2W_ALIGNED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2W_ALIGNED_HIP_THIGH_ACTUATOR_CFG,
    GO2W_ALIGNED_CALF_ACTUATOR_CFG,
    GO2W_ALIGNED_WHEEL_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Wheels get condim 4 (torsional friction: resists the tyre twisting about the
# contact normal when steering in place). Nominal torsional 0.01, randomized in the
# task. No rolling friction: rolling resistance is lumped into joint frictionloss.
ALIGNED_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_wheel_regex: 4, ".*_collision": 1},
  priority={_wheel_regex: 1},
  friction={_wheel_regex: (0.6, 0.01, 0.0)},
  contype=1,
  conaffinity=0,
)

# Aligned-V2 (2026-11-19): wheels condim 6 with rolling friction. Nominal mu_roll
# 0.0013 m (Crr ~0.015 x r 0.086; small pneumatic tyres on hard floors measure Crr
# 0.007-0.026); randomized per wheel in the task. Rolling resistance is load
# dependent and acts at the contacts, so with a multi-point hull wheel it also
# makes yaw torques -- unitree_mujoco's go2w.xml (mu_roll 0.01) showed 50 deg / 12 s
# of heading drift in sim2sim that a v1 policy never corrected.
ALIGNED_V2_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_wheel_regex: 6, ".*_collision": 1},
  priority={_wheel_regex: 1},
  friction={_wheel_regex: (0.6, 0.01, 0.0013)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2W_LEG_ACTUATOR_CFG, GO2W_WHEEL_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2w_robot_cfg() -> EntityCfg:
  """Get a fresh Go2W robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2W_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_go2w_robot_cfg())

  viewer.launch(robot.spec.compile())

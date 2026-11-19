"""Unitree Go2W constants.

MJCF: xmls/go2w.xml -- Unitree's official mjlab go2.xml with wheels added; see the
header comment there for provenance. Link masses follow the go2w_description URDF
that robot_lab trains on.

Actuator gains, limits and init pose mirror robot_lab's UNITREE_GO2W_CFG.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
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

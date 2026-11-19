"""Joint actions that honor `preserve_order`.

mjlab's joint transmission resolves targets via
`Entity.find_joints_by_actuator_names`, which hard-codes preserve_order=False, so
actions always come out in MJCF order. These subclasses reorder the targets to the
exact names listed in `actuator_names` when `preserve_order=True`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs.mdp.actions import (
  JointPositionAction,
  JointPositionActionCfg,
  JointVelocityAction,
  JointVelocityActionCfg,
)


def _reorder(ids: list[int], names: list[str], cfg) -> tuple[list[int], list[str]]:
  if not cfg.preserve_order:
    return ids, names
  rank = {n: i for i, n in enumerate(cfg.actuator_names)}
  missing = [n for n in names if n not in rank]
  assert not missing, f"preserve_order needs exact names, got patterns for {missing}"
  pairs = sorted(zip(ids, names, strict=True), key=lambda p: rank[p[1]])
  return [p[0] for p in pairs], [p[1] for p in pairs]


class OrderedJointPositionAction(JointPositionAction):
  def _find_targets(self, cfg):
    return _reorder(*super()._find_targets(cfg), cfg)


class OrderedJointVelocityAction(JointVelocityAction):
  def _find_targets(self, cfg):
    return _reorder(*super()._find_targets(cfg), cfg)


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env) -> OrderedJointPositionAction:
    return OrderedJointPositionAction(self, env)


@dataclass(kw_only=True)
class OrderedJointVelocityActionCfg(JointVelocityActionCfg):
  def build(self, env) -> OrderedJointVelocityAction:
    return OrderedJointVelocityAction(self, env)

"""MDP terms for the Go2W rough WheelPos task (11-24).

- wheel_vel_from_pos: actor observation, wheel speed from the wheel angle.
- init_rough_terrain_at_top: startup event, resumed runs skip the climb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class wheel_vel_from_pos:
  """[B, 4] wheel speed (rad/s) from the wheel angle: (q_t - q_{t-k}) / (k * step_dt).

  The motor's own dq is an instantaneous estimate (noise std ~0.04 rad/s at rest,
  measured on the real robot 11-24, growing with speed); the encoder angle is good
  to ~3e-5 rad, so a difference over k policy steps is the mean speed over that
  window, almost noise-free, ~k/2 steps late. The wheel angle is not otherwise in
  the actor (joint_pos_rel_without_wheel), so this is new information.

  On reset the window is refilled with the current angle (reads 0 until k steps
  have passed). Deploy must compute the same thing from lowstate q: per step,
  d = q - q_prev, if |d| > 1 rad (wrap / glitch) use dq * step_dt instead, keep the
  last k increments and divide their sum by k * step_dt.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    asset = env.scene["robot"]
    names = list(cfg.params["joint_names"])
    ids, found = asset.find_joints(names, preserve_order=True)
    assert list(found) == names, (found, names)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self._k = int(cfg.params["window"])
    self._dt = env.step_dt
    self._buf = torch.zeros(env.num_envs, self._k + 1, len(ids), device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self._last_step = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._fresh[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    joint_names: tuple[str, ...],
    window: int,
  ) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos[:, self._ids]
    step = int(env.common_step_counter)
    if step != self._last_step:
      # Advance the window once per env step (compute() may run more than once).
      self._buf = torch.roll(self._buf, shifts=-1, dims=1)
      self._last_step = step
    self._buf[:, -1] = q
    if self._fresh.any():
      self._buf[self._fresh] = q[self._fresh].unsqueeze(1)
      self._fresh[:] = False
    return (self._buf[:, -1] - self._buf[:, 0]) / (self._k * self._dt)


def init_rough_terrain_at_top(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  skip_terrains: tuple[str, ...] = ("flat", "random_rough"),
) -> None:
  """Startup: envs on the columns the terrain curriculum moves (not
  ``skip_terrains``) start on the top row, so a resumed run starts where the
  earlier run's curriculum had got to instead of climbing from row 0 again. The
  skipped columns stay on row 0, as in the earlier run. The curriculum then moves
  envs down as usual."""
  terrain = env.scene.terrain
  if terrain.terrain_origins is None:
    return
  names = list(terrain.cfg.terrain_generator.sub_terrains)
  move = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  for i, n in enumerate(names):
    if n in skip_terrains:
      move &= terrain.terrain_types != i
  terrain.terrain_levels[move] = terrain.max_terrain_level - 1
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]

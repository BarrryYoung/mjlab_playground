"""MDP terms for the Go2W rough tasks.

- TerrainAwareBucketVelocityCommand: the flat tasks' bucket command everywhere,
  except on the stair columns: there the heading is fixed for the whole episode
  (the spawn heading) and the robot drives straight forward or backward.
- tracking_curriculum: per-episode velocity tracking score by terrain column;
  logs it and, if ``promote`` is on, moves envs up / down a row on it.
- HeadingRoughVelocityCommand + terrain_levels_vel_on: bucket command on flat,
  ANYmal-rough heading command (no vy) elsewhere, promoted on distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_playground.velocity.config.go2w_climb.climb_mdp import terrain_in
from mjlab_playground.velocity.mdp.commands import (
  BucketVelocityCommand,
  BucketVelocityCommandCfg,
)
from mjlab_playground.velocity.mdp.rewards import (
  _scaled_std,
  diagonal_gait_v2,
  feet_swing_clearance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TerrainAwareBucketVelocityCommand(BucketVelocityCommand):
  """Bucket command, with straight heading-held driving on ``straight_terrains``.

  On those columns every resample is vx = +-U(straight_lin_vel_x), vy = 0, and
  wz comes from a P controller on the heading error each step. The heading
  target is set once per episode, to the heading at spawn (+- noise), so the
  robot keeps one line across the tile.

  Also accumulates, per env and per episode, the velocity tracking kernels the
  tracking curriculum scores on (read in its compute, which runs before reset).
  """

  cfg: TerrainAwareBucketVelocityCommandCfg

  def __init__(self, cfg: TerrainAwareBucketVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.straight = torch.zeros(n, dtype=torch.bool, device=dev)
    self._new_episode = torch.zeros(n, dtype=torch.bool, device=dev)
    # The heading is stale inside a reset (pose written, forward() not run), so
    # the target is taken at the next _update_command.
    self._heading_pending = torch.zeros(n, dtype=torch.bool, device=dev)
    self._heading_noise = torch.zeros(n, device=dev)
    self.track_lin_sum = torch.zeros(n, device=dev)
    self.track_ang_sum = torch.zeros(n, device=dev)
    self.track_steps = torch.zeros(n, device=dev)

  def reset(self, env_ids):
    self._new_episode[env_ids] = True
    extras = super().reset(env_ids)
    self._new_episode[env_ids] = False
    self.track_lin_sum[env_ids] = 0.0
    self.track_ang_sum[env_ids] = 0.0
    self.track_steps[env_ids] = 0.0
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = self.cfg
    mask = terrain_in(self._env, cfg.straight_terrains)[env_ids]
    self.straight[env_ids] = mask
    ids = env_ids[mask]
    if len(ids) == 0:
      return
    r = torch.empty(len(ids), device=self.device)
    vx = r.uniform_(*cfg.straight_lin_vel_x)
    back = torch.rand(len(ids), device=self.device) < cfg.straight_backward_prob
    self.vel_command_b[ids, 0] = torch.where(back, -vx, vx)
    self.vel_command_b[ids, 1:] = 0.0
    self.is_standing_env[ids] = False
    self.vel_command_w[ids] = self.vel_command_b[ids]
    new = ids[self._new_episode[ids]]
    if len(new):
      self._heading_noise[new] = torch.empty(len(new), device=self.device).uniform_(
        -cfg.straight_heading_noise, cfg.straight_heading_noise
      )
      self._heading_pending[new] = True

  def _update_command(self) -> None:
    if self._heading_pending.any():
      ids = self._heading_pending.nonzero(as_tuple=False).flatten()
      self.heading_target[ids] = wrap_to_pi(
        self.robot.data.heading_w[ids] + self._heading_noise[ids]
      )
      self._heading_pending[ids] = False
    super()._update_command()
    ids = self.straight.nonzero(as_tuple=False).flatten()
    if len(ids):
      if not self.cfg.straight_hold_heading:
        # Open loop: the policy itself has to go straight (wz command 0).
        self.vel_command_b[ids, 2] = 0.0
        return
      err = wrap_to_pi(self.heading_target[ids] - self.robot.data.heading_w[ids])
      lim = self.cfg.straight_max_yaw_rate
      self.vel_command_b[ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * err, min=-lim, max=lim
      )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    cfg = self.cfg
    cmd = self.vel_command_b
    lin_err = (cmd[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]).square().sum(1)
    ang_err = (cmd[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]).square()
    self.track_lin_sum += torch.exp(-lin_err / cfg.score_lin_std**2)
    self.track_ang_sum += torch.exp(-ang_err / cfg.score_ang_std**2)
    self.track_steps += 1.0


@dataclass(kw_only=True)
class TerrainAwareBucketVelocityCommandCfg(BucketVelocityCommandCfg):
  straight_terrains: tuple[str, ...] = ("stairs_up", "stairs_down")
  straight_lin_vel_x: tuple[float, float] = (0.3, 1.0)
  straight_backward_prob: float = 0.5
  straight_heading_noise: float = math.radians(10.0)
  straight_max_yaw_rate: float = 1.0
  # False: no heading P controller on the straight columns, wz command stays 0.
  straight_hold_heading: bool = True
  # Kernels of the tracking score: the tracking rewards' std (lin 0.5, ang
  # sqrt(0.5)), fixed rather than command-scaled so the score reads the same
  # for slow and fast commands.
  score_lin_std: float = 0.5
  score_ang_std: float = math.sqrt(0.5)

  def build(self, env) -> TerrainAwareBucketVelocityCommand:
    return TerrainAwareBucketVelocityCommand(self, env)


class HeadingRoughVelocityCommand(TerrainAwareBucketVelocityCommand):
  """Bucket command on flat; ANYmal-rough style heading command elsewhere.

  Off ``bucket_terrains`` every resample is vx = U(heading_lin_vel_x), vy = 0,
  a world heading target U(-pi, pi) tracked by the P controller (wz clipped to
  +-heading_max_yaw_rate), standing with ``heading_standing_prob``, held for
  ``heading_resampling_time`` s. The command is kept long and straight so the
  net distance the terrain curriculum measures means something.
  """

  cfg: HeadingRoughVelocityCommandCfg

  def __init__(self, cfg: HeadingRoughVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self.heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = self.cfg
    mask = ~terrain_in(self._env, cfg.bucket_terrains)[env_ids]
    self.heading_env[env_ids] = mask
    ids = env_ids[mask]
    if len(ids) == 0:
      return
    n, dev = len(ids), self.device
    self.vel_command_b[ids, 0] = torch.empty(n, device=dev).uniform_(
      *cfg.heading_lin_vel_x
    )
    self.vel_command_b[ids, 1:] = 0.0
    self.heading_target[ids] = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
    self.is_standing_env[ids] = torch.rand(n, device=dev) < cfg.heading_standing_prob
    self.vel_command_w[ids] = self.vel_command_b[ids]
    self.time_left[ids] = cfg.heading_resampling_time

  def _update_command(self) -> None:
    super()._update_command()
    ids = (self.heading_env & ~self.is_standing_env).nonzero(as_tuple=False).flatten()
    if len(ids):
      err = wrap_to_pi(self.heading_target[ids] - self.robot.data.heading_w[ids])
      lim = self.cfg.heading_max_yaw_rate
      self.vel_command_b[ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * err, min=-lim, max=lim
      )


@dataclass(kw_only=True)
class HeadingRoughVelocityCommandCfg(TerrainAwareBucketVelocityCommandCfg):
  straight_terrains: tuple[str, ...] = ()
  bucket_terrains: tuple[str, ...] = ("flat",)
  heading_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
  heading_max_yaw_rate: float = 1.0
  heading_standing_prob: float = 0.02
  heading_resampling_time: float = 10.0

  def build(self, env) -> HeadingRoughVelocityCommand:
    return HeadingRoughVelocityCommand(self, env)


def track_linear_velocity_terrain(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  full_terrains: tuple[str, ...] = ("flat",),
  cmd_scaled: bool = False,
  std_min: float = 0.05,
) -> torch.Tensor:
  """track_linear_velocity_scaled, with the vertical speed error and the
  command-scaled std only on ``full_terrains``. On stairs, slopes and rough
  ground the base has to move up and down; charging the world z error there
  costs the term while the robot is climbing, the same way roll / pitch rates
  do in the angular term. And the heading command writes wz every step, mostly
  small: a std scaled down to 0.1 there makes the tolerance far tighter than on
  flat, so those columns keep the fixed std."""
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  full = terrain_in(env, full_terrains)
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2]) * full.float()
  s = torch.where(
    full, _scaled_std(command[:, :2].norm(dim=1), std, cmd_scaled, std_min), std
  )
  return torch.exp(-xy_error / s**2 - z_error / std**2)


def terrain_levels_vel_on(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  skip_terrains: tuple[str, ...] = ("flat",),
  up_score: float = 0.6,
) -> dict[str, torch.Tensor]:
  """terrain_levels_vel (up: net distance > half a tile; down: < half of the
  commanded distance), with the envs on ``skip_terrains`` left where they are:
  the bucket command there (standing, turning in place) does not travel.
  Promotion also needs the episode's tracking score (as tracking_curriculum)
  >= ``up_score``; demotion is on distance only."""
  terrain = env.scene.terrain
  gen = terrain.cfg.terrain_generator
  term = env.command_manager.get_term(command_name)
  command = env.command_manager.get_command(command_name)
  asset = env.scene["robot"]
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )
  n = term.track_steps[env_ids].clamp(min=1)
  score = 0.5 * (term.track_lin_sum[env_ids] + term.track_ang_sum[env_ids]) / n
  move_up = (distance > gen.size[0] / 2) & (score >= up_score)
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  ) & ~move_up
  active = ~terrain_in(env, skip_terrains)[env_ids]
  terrain.update_env_origins(env_ids, move_up & active, move_down & active)
  levels = terrain.terrain_levels.float()
  out = {"mean": levels.mean()}
  for i, name in enumerate(gen.sub_terrains):
    m = terrain.terrain_types == i
    if m.any():
      lv = levels[m]
      out[name] = lv.mean()
      # An env promoted past the top row is put on a random row, so these read
      # the current rows, not how many envs have graduated.
      out[f"{name}_max"] = lv.max()
      out[f"{name}_p50"] = torch.quantile(lv, 0.5)
      out[f"{name}_p90"] = torch.quantile(lv, 0.9)
  return out


def track_angular_velocity_terrain(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  full_terrains: tuple[str, ...] = ("flat",),
  cmd_scaled: bool = False,
  std_min: float = 0.05,
) -> torch.Tensor:
  """track_angular_velocity_scaled, with the roll / pitch rate part and the
  command-scaled std only on ``full_terrains``. Elsewhere only the yaw rate
  error is scored: on stairs, slopes and rough ground the body has to rock, and
  charging it there cost the term ~0.4 of its 1.0 while the yaw rate itself was
  tracked. The heading controller's wz is small most of the time, so a scaled
  std would sit at std_min 0.1 there; those columns keep the fixed std."""
  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  full = terrain_in(env, full_terrains)
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  xy_error = xy_error * full.float()
  s = torch.where(full, _scaled_std(command[:, 2].abs(), std, cmd_scaled, std_min), std)
  return torch.exp(-z_error / s**2 - xy_error / std**2)


class tracking_curriculum:
  """Per-episode tracking score; logs it by column, optionally moves rows on it.

  score = mean over the episode of (lin kernel + ang kernel) / 2, in [0, 1].
  An episode succeeds if it did not fall and score >= ``up_score``; it fails if
  it fell or score < ``down_score``. With ``promote`` on, success moves the env
  one row up and failure one row down (as terrain_levels_vel does, but on how
  well the command was followed instead of the net distance travelled).
  Episodes shorter than ``min_steps`` are not scored.

  Logged under Curriculum/<term>/: <column>_score, <column>_succ, <column>_fell,
  <column>_level, and the same for "all". Values are running means (EMA over
  episodes, ``ema``) so they are not dominated by whichever envs reset this step.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    gen = env.scene.terrain.cfg.terrain_generator
    self.names = list(gen.sub_terrains)
    self.stats: dict[str, torch.Tensor] = {}

  def _ema(self, key: str, value: torch.Tensor, a: float) -> None:
    old = self.stats.get(key)
    self.stats[key] = value if old is None else (1 - a) * old + a * value

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    promote: bool = False,
    up_score: float = 0.6,
    down_score: float = 0.3,
    min_steps: int = 50,
    fell_term: str = "fell_over",
    ema: float = 0.05,
  ) -> dict[str, torch.Tensor]:
    term = env.command_manager.get_term(command_name)
    terrain = env.scene.terrain
    steps = term.track_steps[env_ids]
    valid = steps >= min_steps
    if valid.any():
      ids = env_ids[valid]
      n = steps[valid]
      score = 0.5 * (term.track_lin_sum[ids] + term.track_ang_sum[ids]) / n
      fell = env.termination_manager.get_term(fell_term)[ids]
      succ = ~fell & (score >= up_score)
      fail = fell | (score < down_score)
      if promote:
        terrain.update_env_origins(ids, succ, fail & ~succ)
      cols = terrain.terrain_types[ids]
      a = ema
      for i, name in enumerate(["all", *self.names]):
        m = torch.ones_like(cols, dtype=torch.bool) if i == 0 else cols == i - 1
        if m.any():
          self._ema(f"{name}_score", score[m].mean(), a)
          self._ema(f"{name}_succ", succ[m].float().mean(), a)
          self._ema(f"{name}_fell", fell[m].float().mean(), a)
    levels = terrain.terrain_levels.float()
    out = dict(self.stats)
    out["all_level"] = levels.mean()
    for i, name in enumerate(self.names):
      m = terrain.terrain_types == i
      if m.any():
        out[f"{name}_level"] = levels[m].mean()
    return out


def flat_ground_mask(
  env: ManagerBasedRlEnv, sensor_name: str, max_std: float
) -> torch.Tensor:
  """[B] bool, True where the terrain under the base is flat: the std of the
  terrain_scan hit heights (rays that miss left out) is below ``max_std`` m."""
  sensor = env.scene[sensor_name]
  data = sensor.data
  b = data.distances.shape[0]
  z = data.hit_pos_w[..., 2].view(b, -1)
  valid = (data.distances.view(b, -1) >= 0).float()
  n = valid.sum(1).clamp(min=1.0)
  mean = (z * valid).sum(1) / n
  var = (((z - mean[:, None]) ** 2) * valid).sum(1) / n
  return var.sqrt() < max_std


class diagonal_gait_v2_gated(diagonal_gait_v2):
  """dgv2 paid on the ``posture_terrains`` columns, and also wherever the ground
  under the base is flat (scan std < ``flat_std``), e.g. a stair tile's centre."""

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    posture_terrains: tuple[str, ...],
    scan_sensor_name: str,
    flat_std: float,
    **kw,
  ):
    on = terrain_in(env, posture_terrains) | flat_ground_mask(
      env, scan_sensor_name, flat_std
    )
    return super().__call__(env, **kw) * on.float()


class feet_swing_clearance_gated(feet_swing_clearance):
  """feet_swing_clearance with the same gate as diagonal_gait_v2_gated."""

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    posture_terrains: tuple[str, ...],
    scan_sensor_name: str,
    flat_std: float,
    **kw,
  ):
    on = terrain_in(env, posture_terrains) | flat_ground_mask(
      env, scan_sensor_name, flat_std
    )
    return super().__call__(env, **kw) * on.float()


def _spawn_heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B] world heading at the start of the episode, latched on the first steps.

  The reward manager runs before the reset of a step, the observation manager
  after it, so a fresh env is seen with episode_length_buf 0 (obs) and 1
  (reward): latch on <= 1, shared by the reward and the critic observation."""
  heading = env.scene["robot"].data.heading_w
  if not hasattr(env, "_spawn_heading_ref"):
    env._spawn_heading_ref = heading.clone()
  ref = env._spawn_heading_ref
  fresh = env.episode_length_buf <= 1
  if fresh.any():
    ref[fresh] = heading[fresh]
  return ref


def heading_drift_penalty(
  env: ManagerBasedRlEnv, terrains: tuple[str, ...], tolerance: float = 0.8
) -> torch.Tensor:
  """min(|heading - spawn heading| / tolerance, 1) on ``terrains``, else 0.

  Use with a negative weight. A pure penalty, bounded by the weight: standing
  still never pays more than walking a little crooked."""
  err = wrap_to_pi(env.scene["robot"].data.heading_w - _spawn_heading(env)).abs()
  return torch.clamp(err / tolerance, max=1.0) * terrain_in(env, terrains).float()


def heading_error_from_spawn(
  env: ManagerBasedRlEnv, terrains: tuple[str, ...]
) -> torch.Tensor:
  """[B, 1] signed heading error from the spawn heading (rad) on ``terrains``,
  0 elsewhere. Critic only: the actor cannot see the spawn heading."""
  err = wrap_to_pi(env.scene["robot"].data.heading_w - _spawn_heading(env))
  return (err * terrain_in(env, terrains).float()).unsqueeze(-1)

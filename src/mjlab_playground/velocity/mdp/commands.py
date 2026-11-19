"""Velocity commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class SignedForwardVelocityCommand(UniformVelocityCommand):
  """Forward-only envs drive straight forward *or* backward, half each.

  Upstream makes them |vx| clamped to >= 0.3, i.e. forward only. Here the sign
  is flipped on a random half after the upstream resample, so the magnitude
  distribution is unchanged and vy / wz stay zero.
  """

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      back = torch.rand(len(fwd_ids), device=self.device) < 0.5
      self.vel_command_b[fwd_ids[back], 0] *= -1.0
      # init_velocity_prob would have seeded the root with the unflipped vx.
      assert self.cfg.init_velocity_prob == 0.0


@dataclass(kw_only=True)
class SignedForwardVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> SignedForwardVelocityCommand:
    return SignedForwardVelocityCommand(self, env)


class GroupedVelocityCommand(UniformVelocityCommand):
  """Every resample puts each env in exactly one group, with fixed probabilities.

  Upstream rolls standing / heading / forward as independent dice, so groups
  overlap and the effective split has to be worked out by hand. Here they are
  mutually exclusive, and a low-speed group is carved out of the uniform one so
  slow commands (e.g. wz = 0.05) actually get trained on.

  - standing: zero command.
  - heading:  vx, vy uniform; wz from the heading error (upstream logic).
  - forward:  |vx| uniform clamped to >= 0.3, random sign; vy = wz = 0.
  - low:      vx, vy, wz uniform in ``low_ranges``, each then zeroed with prob
              ``low_zero_prob``, so pure slow yaw / slow drive show up.
  - uniform:  the rest; vx, vy, wz uniform in the full ranges.

  The rel_* fields of the base cfg are ignored.
  """

  cfg: GroupedVelocityCommandCfg

  def __init__(self, cfg: GroupedVelocityCommandCfg, env) -> None:
    total = cfg.p_standing + cfg.p_heading + cfg.p_forward + cfg.p_low
    assert total <= 1.0, f"group probabilities sum to {total} > 1"
    # init_velocity_prob would seed the root from the pre-group command.
    assert cfg.init_velocity_prob == 0.0
    super().__init__(cfg, env)
    self.is_low_env = torch.zeros_like(self.is_heading_env)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*cfg.ranges.ang_vel_z)
    if cfg.heading_command:
      assert cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*cfg.ranges.heading)

    edges = torch.tensor(
      [cfg.p_standing, cfg.p_heading, cfg.p_forward, cfg.p_low], device=self.device
    ).cumsum(0)
    # 0 standing, 1 heading, 2 forward, 3 low, 4 uniform
    group = torch.bucketize(torch.rand(n, device=self.device), edges, right=True)
    self.is_standing_env[env_ids] = group == 0
    self.is_heading_env[env_ids] = (group == 1) & cfg.heading_command
    self.is_forward_env[env_ids] = group == 2
    self.is_low_env[env_ids] = group == 3
    self.is_world_env[env_ids] = False

    fwd_ids = env_ids[group == 2]
    if len(fwd_ids) > 0:
      vx = self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      back = torch.rand(len(fwd_ids), device=self.device) < 0.5
      self.vel_command_b[fwd_ids, 0] = torch.where(back, -vx, vx)
      self.vel_command_b[fwd_ids, 1:] = 0.0

    low_ids = env_ids[group == 3]
    if len(low_ids) > 0:
      lo = torch.empty(len(low_ids), 3, device=self.device)
      lo[:, 0].uniform_(*cfg.low_ranges.lin_vel_x)
      lo[:, 1].uniform_(*cfg.low_ranges.lin_vel_y)
      lo[:, 2].uniform_(*cfg.low_ranges.ang_vel_z)
      keep = torch.rand(len(low_ids), 3, device=self.device) >= cfg.low_zero_prob
      self.vel_command_b[low_ids] = lo * keep

    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@dataclass(kw_only=True)
class GroupedVelocityCommandCfg(UniformVelocityCommandCfg):
  # Defaults keep the effective split of the upstream independent dice
  # (standing 10, heading 21.6, forward 12.6 + heading&forward 5.4,
  # uniform 50.4) and move 15 points of uniform into the low group.
  p_standing: float = 0.10
  p_heading: float = 0.22
  p_forward: float = 0.18
  p_low: float = 0.15
  low_ranges: UniformVelocityCommandCfg.Ranges = field(
    default_factory=lambda: UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.4, 0.4)
    )
  )
  low_zero_prob: float = 0.5

  def build(self, env) -> GroupedVelocityCommand:
    return GroupedVelocityCommand(self, env)


class BucketVelocityCommand(UniformVelocityCommand):
  """Mutually exclusive command buckets aimed at the gaits we test.

  - standing: zero command.
  - vx / vy / wz: that axis only, |v| uniform in [``slow_max``, axis max],
    random sign (vx covers both forward and backward).
  - slow: one axis only, picked uniformly among vx / vy / wz, |v| uniform in
    ``slow_speed``, random sign. Straddles the swing gate, so slow steering
    trains both "keep the wheels down" and "step".
  - combo: vx, vy, wz all uniform in the full ranges.

  Heading control is not used; the rel_* fields of the base cfg are ignored.
  """

  cfg: BucketVelocityCommandCfg

  def __init__(self, cfg: BucketVelocityCommandCfg, env) -> None:
    total = sum(cfg.p.values())
    assert abs(total - 1.0) < 1e-6, f"bucket probabilities sum to {total}, not 1"
    assert set(cfg.p) == set(BUCKETS), f"buckets must be {BUCKETS}"
    assert not cfg.heading_command and cfg.init_velocity_prob == 0.0
    super().__init__(cfg, env)
    self.bucket = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    n = len(env_ids)
    dev = self.device
    probs = torch.tensor([cfg.p[b] for b in BUCKETS], device=dev)
    bucket = torch.bucketize(
      torch.rand(n, device=dev), probs.cumsum(0)[:-1], right=True
    )
    self.bucket[env_ids] = bucket

    ranges = (cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z)
    combo = torch.stack(
      [torch.empty(n, device=dev).uniform_(lo, hi) for lo, hi in ranges], dim=1
    )
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    u = torch.rand(n, device=dev)

    # Single-axis buckets: which axis, and the speed band it is drawn from.
    axis = torch.full((n,), -1, dtype=torch.long, device=dev)
    lo = torch.zeros(n, device=dev)
    hi = torch.zeros(n, device=dev)
    for i, name in enumerate(("vx", "vy", "wz")):
      m = bucket == BUCKETS.index(name)
      axis[m] = i
      lo[m] = cfg.slow_speed[1]
      hi[m] = ranges[i][1]
    m = bucket == BUCKETS.index("slow")
    axis[m] = torch.randint(0, 3, (int(m.sum()),), device=dev)
    lo[m], hi[m] = cfg.slow_speed

    cmd = torch.zeros(n, 3, device=dev)
    single = axis >= 0
    cmd[single, axis[single]] = (sign * (lo + u * (hi - lo)))[single]
    is_combo = bucket == BUCKETS.index("combo")
    cmd[is_combo] = combo[is_combo]
    if cfg.deadzone > 0.0:
      cmd = torch.where(cmd.abs() < cfg.deadzone, 0.0, cmd)

    self.vel_command_b[env_ids] = cmd
    self.is_standing_env[env_ids] = bucket == BUCKETS.index("standing")
    self.is_heading_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.vel_command_w[env_ids] = cmd


BUCKETS = ("standing", "vx", "vy", "wz", "slow", "combo")


@dataclass(kw_only=True)
class BucketVelocityCommandCfg(UniformVelocityCommandCfg):
  p: dict[str, float] = field(
    default_factory=lambda: {
      "standing": 0.10,
      "vx": 0.10,
      "vy": 0.10,
      "wz": 0.10,
      "slow": 0.30,
      "combo": 0.30,
    }
  )
  # Slow single-axis band; its top is also where the normal single-axis band starts.
  slow_speed: tuple[float, float] = (0.05, 0.3)
  # Per-axis deadzone: |v| below it is set to 0, as a joystick deadzone on the
  # robot would. 0 keeps every sampled value.
  deadzone: float = 0.0

  def build(self, env) -> BucketVelocityCommand:
    return BucketVelocityCommand(self, env)

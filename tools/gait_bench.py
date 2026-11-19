"""Gait benchmark for Go2W checkpoints under fixed commands.

For every checkpoint, each command gets 32 envs; the policy runs
deterministically, the first 2 s are dropped and the next 6 s are measured.

  * zero command: mean |wheel speed| (creeping) and how often a wheel lifts.
  * vy / wz commands, per wheel: air time, contact time and swing height,
    each as mean and 90% interval [p5, p95].

Durations come from the contact sensor's own air-time tracking, which advances
every physics substep, so they have 5 ms resolution. A phase is recorded when
it ends; a phase still running when the window closes is recorded too, as a
lower bound, so a wheel hanging for the whole window still shows up. Swing
height is the peak ground clearance (wheel-centre height minus radius) seen at
policy steps during that air phase.

    uv run python tools/gait_bench.py logs/rsl_rl/<exp>/<run> [more paths ...]

Prints 22 pass/fail tests per checkpoint (thresholds at the top of the file).
For detail: --table prints per-wheel numbers, --plot draws per-wheel range
plots of all checkpoints on shared axes.

Paths resolve like `rewardlab`: experiment dir -> newest run, run dir -> latest
checkpoint, or a model_*.pt file. --task picks the task when several share a
log dir.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_playground.reward_lab import _resolve_path

COMMANDS = [
  ("zero", (0.0, 0.0, 0.0)),
  ("vy0.15", (0.0, 0.15, 0.0)),
  ("vy0.25", (0.0, 0.25, 0.0)),
  ("vy0.5", (0.0, 0.5, 0.0)),
  ("wz0.15", (0.0, 0.0, 0.15)),
  ("wz0.5", (0.0, 0.0, 0.5)),
  ("wz1.0", (0.0, 0.0, 1.0)),
  ("wz2.0", (0.0, 0.0, 2.0)),
]
ENVS_PER_CMD = 32
WARMUP_S = 2.0
MEASURE_S = 6.0
DEVICE = "cuda:0"

# Pass/fail (targets set by --wheel-speed / --height / --time). Test 1:
# zero-command wheel speed <= the limit. Test 2: for each stepping command and
# metric, the four wheels' [p5, p95] averaged must sit inside
# [target / 2, target * 1.5].


def _load_policy(ckpt: Path):
  sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)["actor_state_dict"]
  if any("normalizer" in k for k in sd):
    raise SystemExit(f"{ckpt}: actor has an obs normalizer, not handled here")
  idx = sorted(
    {int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith("weight")}
  )
  layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in idx]

  def policy(x: torch.Tensor) -> torch.Tensor:
    for i, (w, b) in enumerate(layers):
      x = x @ w.T + b
      if i < len(layers) - 1:
        x = torch.nn.functional.elu(x)
    return x

  return policy


def _make_env(task: str) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = len(COMMANDS) * ENVS_PER_CMD
  # Building the env prints every manager's term table; keep the report readable.
  with contextlib.redirect_stdout(io.StringIO()):
    return ManagerBasedRlEnv(cfg, device=DEVICE)


def _pin_commands(env: ManagerBasedRlEnv) -> None:
  """Hold every env on its fixed command: no resampling, standing or heading."""
  table = torch.tensor(
    [c for _, c in COMMANDS for _ in range(ENVS_PER_CMD)],
    dtype=torch.float32,
    device=DEVICE,
  )
  term = env.command_manager.get_term("twist")

  def fixed(self, env_ids):
    self.vel_command_b[env_ids] = table[env_ids]
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    if hasattr(self, "is_low_env"):
      self.is_low_env[env_ids] = False

  term._resample_command = types.MethodType(fixed, term)
  term._resample_command(torch.arange(len(table), device=DEVICE))


def _clearance_fn(env: ManagerBasedRlEnv):
  """Per-wheel ground clearance in contact-sensor order, plus wheel labels."""
  rew_cfg = env.cfg.rewards["foot_swing_clearance"]
  aligned = rew_cfg.func(rew_cfg, env)  # owns the contact->height-sensor permutation
  height = env.scene[rew_cfg.params["height_sensor_name"]]
  contact = env.scene[rew_cfg.params["sensor_name"]]
  labels = [
    s.primary_name.split("_")[0] for s in contact._slots if s.field_name == "found"
  ]
  return (lambda: height.data.heights[:, aligned.perm] - aligned.radius), labels


def _stats(x: torch.Tensor, scale: float = 1.0) -> tuple[float, float, float] | None:
  """(mean, p5, p95), or None when there is nothing to summarise."""
  if x.numel() == 0:
    return None
  q = torch.quantile(x.float(), torch.tensor([0.05, 0.95], device=x.device))
  return tuple(v * scale for v in (x.float().mean().item(), q[0].item(), q[1].item()))


@dataclass
class Result:
  name: str
  labels: list[str]
  wheel_speed: float
  zero_landings: float  # per wheel per second under the zero command
  falls: int
  # Over the stepping commands, policy steps with at least one wheel in the air,
  # and among those, steps where exactly one diagonal pair is on the ground.
  some_air_steps: int = 0
  diag_steps: tuple[int, int] = (0, 0)  # (FL+RR only, FR+RL only)
  # (command index, wheel index) -> (mean, p5, p95) per metric
  air: dict = field(default_factory=dict)
  contact: dict = field(default_factory=dict)
  height: dict = field(default_factory=dict)


def bench(env: ManagerBasedRlEnv, ckpt: Path) -> Result:
  policy = _load_policy(ckpt)
  contact = env.scene[env.cfg.rewards["foot_swing_clearance"].params["sensor_name"]]
  clearance, labels = _clearance_fn(env)
  robot = env.scene["robot"]
  wheel_ids, _ = robot.find_joints(".*_wheel_joint")
  # A phase that ended during this policy step has its current timer below one
  # step (half a substep of slack; see the float32 note in soft_landing_peak).
  window = env.step_dt + 0.5 * env.physics_dt

  obs, _ = env.reset()
  _pin_commands(env)
  n_env = env.num_envs
  group = torch.arange(n_env, device=DEVICE) // ENVS_PER_CMD
  peak = torch.zeros(n_env, 4, device=DEVICE)
  # Each record: (command index, wheel index, value), concatenated at the end.
  air, cont, height = [], [], []
  wheel_speed, falls = [], torch.zeros(n_env, device=DEVICE)

  # Diagonal pairs by wheel name; contact slots follow MJCF order.
  diag_a = torch.tensor([labels.index("FL"), labels.index("RR")], device=DEVICE)
  diag_b = torch.tensor([labels.index("FR"), labels.index("RL")], device=DEVICE)
  stepping = group != 0
  some_air_steps, diag_counts = 0, [0, 0]

  def record(store, mask, values):
    e, f = mask.nonzero(as_tuple=True)
    store.append(torch.stack([group[e].float(), f.float(), values[mask]], dim=1))

  warmup = round(WARMUP_S / env.step_dt)
  total = warmup + round(MEASURE_S / env.step_dt)
  for t in range(total):
    with torch.no_grad():
      obs, _, terminated, _, _ = env.step(policy(obs["actor"]))
    d = contact.data
    in_air = d.current_air_time > 0
    peak = torch.where(in_air, torch.maximum(peak, clearance()), peak)
    landed = (d.current_contact_time > 0) & (d.current_contact_time < window)
    lifted = (d.current_air_time > 0) & (d.current_air_time < window)

    if t >= warmup:
      # Only the part of a phase inside the window counts: a phase that began
      # during warmup is clipped to the time since the window opened.
      since_start = (t - warmup + 1) * env.step_dt
      falls += terminated.float()
      down = ~in_air
      some_air = stepping & in_air.any(dim=1)
      only_a = down[:, diag_a].all(1) & ~down[:, diag_b].any(1)
      only_b = down[:, diag_b].all(1) & ~down[:, diag_a].any(1)
      some_air_steps += int(some_air.sum())
      diag_counts[0] += int((stepping & only_a).sum())
      diag_counts[1] += int((stepping & only_b).sum())
      wheel_speed.append(robot.data.joint_vel[:, wheel_ids].abs())
      record(air, landed, d.last_air_time.clamp(max=since_start))
      record(height, landed, peak)
      record(cont, lifted, d.last_contact_time.clamp(max=since_start))
    peak = torch.where(in_air, peak, torch.zeros_like(peak))

  # Phases still running at the end, recorded as lower bounds.
  d = contact.data
  still_air = d.current_air_time > 0
  record(air, still_air, d.current_air_time.clamp(max=MEASURE_S))
  record(height, still_air, peak)
  record(cont, d.current_contact_time > 0, d.current_contact_time.clamp(max=MEASURE_S))
  air, cont, height = (torch.cat(s) for s in (air, cont, height))

  def gather(store, k, f):
    return store[(store[:, 0] == k) & (store[:, 1] == f), 2]

  ws = torch.stack(wheel_speed)[:, :ENVS_PER_CMD].mean().item()
  lifts = sum(gather(air, 0, f).numel() for f in range(4)) - int(
    still_air[:ENVS_PER_CMD].sum()
  )
  res = Result(
    name=f"{ckpt.parent.name} / {ckpt.stem}",
    labels=labels,
    wheel_speed=ws,
    zero_landings=lifts / (ENVS_PER_CMD * 4 * MEASURE_S),
    falls=int(falls.sum().item()),
    some_air_steps=some_air_steps,
    diag_steps=tuple(diag_counts),
  )
  for k in range(len(COMMANDS)):
    for f in range(4):
      res.air[k, f] = _stats(gather(air, k, f))
      res.contact[k, f] = _stats(gather(cont, k, f))
      res.height[k, f] = _stats(gather(height, k, f), 100.0)
  return res


# ── output ──────────────────────────────────────────────────────────

# (title, Result field, axis max); values past the axis end in an arrow.
PANELS = [
  ("air time s", "air", 0.4),
  ("contact time s", "contact", 0.4),
  ("swing height cm", "height", 5.0),
]
BAR_W = 24


def _bar(stat: tuple[float, float, float] | None, axis_max: float) -> str:
  """One row of a range plot: ├──●──┤ spans [p5, p95], ● is the mean."""
  if stat is None:
    return "(none)".ljust(BAR_W)
  m, lo, hi = stat

  def col(v: float) -> int:
    return max(0, min(BAR_W - 1, round(v / axis_max * (BAR_W - 1))))

  line = [" "] * BAR_W
  a, b = col(lo), col(hi)
  for c in range(a, b + 1):
    line[c] = "─"
  line[a], line[b] = "├", "┤"
  line[col(m)] = "●"
  if hi > axis_max:
    line[b] = "▶"  # an off-axis mean is drawn as ▶ too
  return "".join(line)


def _axis(axis_max: float) -> str:
  right = f"{axis_max:g}"
  return "0" + " " * (BAR_W - 1 - len(right)) + right


def render(results: list[Result]) -> None:
  tags = [chr(ord("A") + i) for i in range(len(results))]
  print()
  for tag, r in zip(tags, results, strict=True):
    print(f"{tag}  {r.name}")
  print("\n● mean   ├──┤ 90% interval [p5, p95]   ▶ runs past the axis\n")
  print("zero command:")
  for tag, r in zip(tags, results, strict=True):
    print(
      f"  {tag}  wheel speed {r.wheel_speed:.3f} rad/s   lifts {r.zero_landings:.2f} /wheel/s"
    )

  gap = "   "
  for k, (cmd, _) in enumerate(COMMANDS):
    if cmd == "zero":
      continue
    print(f"\n── {cmd} " + "─" * (8 + 3 * (BAR_W + len(gap)) - len(cmd)))
    print(" " * 8 + gap.join(t.ljust(BAR_W) for t, _, _ in PANELS))
    print(" " * 8 + gap.join(_axis(m) for _, _, m in PANELS))
    for tag, r in zip(tags, results, strict=True):
      for f in range(4):
        bars = gap.join(_bar(getattr(r, key)[k, f], m) for _, key, m in PANELS)
        print(f"{tag if f == 0 else ' '} {r.labels[f]:4}  {bars}")

  for tag, r in zip(tags, results, strict=True):
    if r.falls:
      print(f"\nnote: {tag} had {r.falls} episode terminations during the window")


def _mark(ok: bool) -> str:
  if not sys.stdout.isatty():
    return "PASS" if ok else "FAIL"
  return "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"


def _wheel_mean(stats: list) -> tuple[float, float]:
  """Average [p5, p95] over the wheels; a wheel with no phase counts as 0."""
  lo = sum(st[1] if st else 0.0 for st in stats) / len(stats)
  hi = sum(st[2] if st else 0.0 for st in stats) / len(stats)
  return lo, hi


def _summary(r: Result) -> str:
  """One line over the stepping commands (zero command excluded).

  air / contact / height: mean over commands and wheels.
  trot: of the time some wheel is in the air, the share where exactly one
  diagonal pair (FL+RR or FR+RL) is on the ground; all-four-down time is left
  out so standing and scrubbing cannot score.
  balance: time on the FL+RR pair vs the FR+RL pair, smaller / larger
  (1 = the two diagonals take equal turns).
  """
  steps = [k for k, (cmd, _) in enumerate(COMMANDS) if cmd != "zero"]

  def avg(store):
    vals = [store[k, f][0] if store[k, f] else 0.0 for k in steps for f in range(4)]
    return sum(vals) / len(vals)

  a, b = r.diag_steps
  trot = (a + b) / r.some_air_steps if r.some_air_steps else 0.0
  balance = min(a, b) / max(a, b) if max(a, b) else 0.0
  return (
    f"summary  air {avg(r.air):.3f} s   contact {avg(r.contact):.3f} s   "
    f"height {avg(r.height):.1f} cm   trot {100 * trot:.0f}%   balance {balance:.2f}"
  )


def render_tests(
  results: list[Result], max_wheel_speed: float, height: float, time: float
) -> None:
  targets = [("height", "cm", height), ("air", "s", time), ("contact", "s", time)]
  print(
    f"\nthresholds  wheel <= {max_wheel_speed:g} rad/s (--wheel-speed, default 0.1)   "
    f"height {height:g} cm -> [{height / 2:g}, {height * 1.5:g}] (--height, default 2)   "
    f"time {time:g} s -> [{time / 2:g}, {time * 1.5:g}] (--time, default 0.2)"
  )

  def row(n, cmd, metric, value, need, ok):
    print(f"{n:>3}  {cmd:7} {metric:8} {value:22} need {need:18} {_mark(ok)}")

  for r in results:
    print(f"\n=== {r.name}")
    passed = 0
    ok = r.wheel_speed <= max_wheel_speed
    passed += ok
    row(1, "zero", "wheel", f"{r.wheel_speed:.3f} rad/s", f"<= {max_wheel_speed:g}", ok)
    n = 1
    for key, unit, target in targets:
      for k, (cmd, _) in enumerate(COMMANDS):
        if cmd == "zero":
          continue
        n += 1
        lo, hi = _wheel_mean([getattr(r, key)[k, f] for f in range(4)])
        need_lo, need_hi = target / 2, target * 1.5
        ok = lo >= need_lo and hi <= need_hi
        passed += ok
        d = 1 if unit == "cm" else 3
        row(
          n,
          cmd,
          key,
          f"[{lo:.{d}f}, {hi:.{d}f}] {unit}",
          f"[{need_lo:.{d}f}, {need_hi:.{d}f}]",
          ok,
        )
    print(f"passed {passed}/{n}")
    print(_summary(r))
    if r.falls:
      print(f"note: {r.falls} episode terminations during the window")


def render_table(results: list[Result]) -> None:
  def fmt(stat, digits):
    if stat is None:
      return "—".ljust(26)
    m, lo, hi = stat
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]".ljust(26)

  for r in results:
    print(f"\n=== {r.name}")
    print(
      f"zero   wheel speed {r.wheel_speed:.3f} rad/s   lifts {r.zero_landings:.2f} /wheel/s"
    )
    print(
      f"{'cmd':6} {'wheel':5} {'air time s  mean [p5, p95]':26} "
      f"{'contact time s':26} {'swing height cm':26}"
    )
    for k, (cmd, _) in enumerate(COMMANDS):
      if cmd == "zero":
        continue
      for f in range(4):
        print(
          f"{cmd:6} {r.labels[f]:5} {fmt(r.air[k, f], 3)} "
          f"{fmt(r.contact[k, f], 3)} {fmt(r.height[k, f], 1)}"
        )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument(
    "paths",
    nargs="+",
    metavar="PATH [WHEEL_SPEED HEIGHT TIME]",
    help="log paths, optionally followed by the three thresholds, e.g. ... 0.1 2 0.2",
  )
  parser.add_argument("--task", default=None)
  parser.add_argument("--table", action="store_true", help="per-wheel numbers")
  parser.add_argument("--plot", action="store_true", help="per-wheel range plots")
  parser.add_argument(
    "--wheel-speed",
    type=float,
    default=0.1,
    help="zero-command wheel speed limit, rad/s (default 0.1)",
  )
  parser.add_argument(
    "--height", type=float, default=2.0, help="swing height target, cm (default 2)"
  )
  parser.add_argument(
    "--time",
    type=float,
    default=0.2,
    help="air and contact time target, s (default 0.2)",
  )
  args = parser.parse_args()

  # Trailing numbers that are not paths are the thresholds, in the order
  # wheel speed, height, time.
  def is_number(x: str) -> bool:
    try:
      float(x)
    except ValueError:
      return False
    return not Path(x).exists()

  numbers = []
  while args.paths and is_number(args.paths[-1]):
    numbers.insert(0, float(args.paths.pop()))
  if numbers:
    if len(numbers) != 3:
      parser.error(f"expected 3 thresholds (wheel speed, height, time), got {numbers}")
    args.wheel_speed, args.height, args.time = numbers
  if not args.paths:
    parser.error("no log path given")

  by_task: dict[str, list[Path]] = defaultdict(list)
  for p in args.paths:
    task, ckpt = _resolve_path(p, args.task)
    by_task[task].append(ckpt)

  results = []
  for task, ckpts in by_task.items():
    print(f"[gait_bench] {task}: {len(ckpts)} checkpoint(s)", flush=True)
    env = _make_env(task)
    for ckpt in ckpts:
      results.append(bench(env, ckpt))
    env.close()

  if args.table:
    render_table(results)
  elif args.plot:
    render(results)
  else:
    render_tests(results, args.wheel_speed, args.height, args.time)


if __name__ == "__main__":
  main()

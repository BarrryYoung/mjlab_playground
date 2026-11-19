#!/usr/bin/env python3
"""Go1 velocity reward ablation sweep.

Sweeps each non-task reward term over 4 weights while holding everything else
at its default. Runs serially; ~14.4 min per run at 1024 envs / 2500 iters.

  uv run python tools/sweep.py                # full sweep, ~10.8 h
  uv run python tools/sweep.py --iters 30     # smoke test, verifies every flag
  uv run python tools/sweep.py --dry-run      # print commands, run nothing

Already-finished runs are skipped, so re-invoking resumes where it stopped.

Weights are swept on a contribution ladder (off -> ~0.02 -> ~0.2 -> ~1.0),
derived from each term's measured raw value rather than by scaling the default.
The track terms contribute ~2.0 total, which is the reference scale.

"Off" is 1e-6 rather than 0.0 because RewardManager.compute() skips terms whose
weight is exactly 0.0, which would drop that term's Episode_Reward curve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASK = "Playground-Velocity-Flat-Unitree-Go1"
EXP = "go1_vel_ablation"
NUM_ENVS = 1024
ITERS = 2500

OFF_P = "1e-6"  # "off" for positive-weight terms
OFF_N = "-1e-6"  # "off" for negative-weight terms

# (run_name_suffix, [(reward_term, weight), ...])
# Ordered by expected information content: the strong terms come first, so an
# early stop still leaves complete curves for what matters most.
SWEEP: list[tuple[str, list[tuple[str, str]]]] = [
  ("base", []),
  # -- action_rate_l2: default -0.1, raw 3.41 --------------------------------
  ("actrate_off", [("action_rate_l2", OFF_N)]),
  ("actrate_0.03", [("action_rate_l2", "-0.03")]),
  ("actrate_0.3", [("action_rate_l2", "-0.3")]),
  ("actrate_1.0", [("action_rate_l2", "-1.0")]),
  # -- upright: default 1.0, raw 0.96 ----------------------------------------
  ("up_off", [("upright", OFF_P)]),
  ("up_0.2", [("upright", "0.2")]),
  ("up_2.0", [("upright", "2.0")]),
  ("up_4.0", [("upright", "4.0")]),
  # -- pose: default 1.0, raw 0.49 -------------------------------------------
  ("pose_off", [("pose", OFF_P)]),
  ("pose_0.4", [("pose", "0.4")]),
  ("pose_2.0", [("pose", "2.0")]),
  ("pose_4.0", [("pose", "4.0")]),
  # -- foot_clearance: default -2.0, raw 0.030 -------------------------------
  ("clear_off", [("foot_clearance", OFF_N)]),
  ("clear_0.5", [("foot_clearance", "-0.5")]),
  ("clear_7", [("foot_clearance", "-7")]),
  ("clear_30", [("foot_clearance", "-30")]),
  # -- air_time: default 0.0 (off), raw ~1.9 ---------------------------------
  ("air_0.05", [("air_time", "0.05")]),
  ("air_0.15", [("air_time", "0.15")]),
  ("air_0.4", [("air_time", "0.4")]),
  ("air_1.0", [("air_time", "1.0")]),
  # -- foot_slip: default -0.1, raw 0.021 ------------------------------------
  ("slip_off", [("foot_slip", OFF_N)]),
  ("slip_1", [("foot_slip", "-1")]),
  ("slip_10", [("foot_slip", "-10")]),
  ("slip_50", [("foot_slip", "-50")]),
  # -- foot_swing_height: default -0.25, raw 0.024 ---------------------------
  ("swing_off", [("foot_swing_height", OFF_N)]),
  ("swing_2", [("foot_swing_height", "-2")]),
  ("swing_8", [("foot_swing_height", "-8")]),
  ("swing_40", [("foot_swing_height", "-40")]),
  # -- soft_landing: default -1e-5, raw 10.0 ---------------------------------
  ("land_off", [("soft_landing", OFF_N)]),
  ("land_0.002", [("soft_landing", "-0.002")]),
  ("land_0.02", [("soft_landing", "-0.02")]),
  ("land_0.1", [("soft_landing", "-0.1")]),
  # -- dof_pos_limits: default -1.0, raw 0.002 -------------------------------
  ("doflim_off", [("dof_pos_limits", OFF_N)]),
  ("doflim_10", [("dof_pos_limits", "-10")]),
  ("doflim_100", [("dof_pos_limits", "-100")]),
  ("doflim_500", [("dof_pos_limits", "-500")]),
  # -- body_ang_vel: default 0.0 (off), raw unknown --------------------------
  ("bodyang_0.01", [("body_ang_vel", "-0.01")]),
  ("bodyang_0.05", [("body_ang_vel", "-0.05")]),
  ("bodyang_0.2", [("body_ang_vel", "-0.2")]),
  ("bodyang_1.0", [("body_ang_vel", "-1.0")]),
  # -- angular_momentum: default 0.0 (off), raw unknown ----------------------
  ("angmom_0.05", [("angular_momentum", "-0.05")]),
  ("angmom_0.5", [("angular_momentum", "-0.5")]),
  ("angmom_2", [("angular_momentum", "-2")]),
  ("angmom_10", [("angular_momentum", "-10")]),
]


def build_cmd(run_name: str, overrides: list[tuple[str, str]], iters: int) -> list[str]:
  cmd = [
    "uv", "run", "train", TASK,
    "--env.scene.num-envs", str(NUM_ENVS),
    "--agent.max-iterations", str(iters),
    "--agent.experiment-name", EXP,
    "--agent.run-name", run_name,
    "--agent.logger", "tensorboard",
  ]  # fmt: skip
  for term, weight in overrides:
    cmd += [f"--env.rewards.{term.replace('_', '-')}.weight", weight]
  return cmd


def is_done(run_name: str, iters: int) -> bool:
  """A run counts as finished once its final checkpoint exists.

  rsl_rl numbers checkpoints by iteration index, so a 2500-iteration run ends
  at model_2499.pt -- there is no model_2500.pt.
  """
  exp_dir = REPO / "logs" / "rsl_rl" / EXP
  if not exp_dir.is_dir():
    return False
  return any(
    (d / f"model_{iters - 1}.pt").exists()
    for d in exp_dir.iterdir()
    if d.is_dir() and d.name.endswith(f"_{run_name}")
  )


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=ITERS)
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--only", default="", help="substring filter on run name")
  args = ap.parse_args()

  runs = [(n, o) for n, o in SWEEP if args.only in n]
  log_dir = REPO / "logs" / "sweep_logs"
  log_dir.mkdir(parents=True, exist_ok=True)

  print(f"sweep: {len(runs)} runs x {args.iters} iters -> logs/rsl_rl/{EXP}/")
  print(f"logs:  {log_dir}\n", flush=True)

  t_start = time.time()
  failed: list[str] = []

  for i, (name, overrides) in enumerate(runs, 1):
    run_name = f"abl_{name}"
    cmd = build_cmd(run_name, overrides, args.iters)
    head = f"[{i}/{len(runs)}] {run_name}"

    if args.dry_run:
      print(f"{head}\n    {' '.join(cmd)}", flush=True)
      continue

    if is_done(run_name, args.iters):
      print(f"{head}  SKIP (already finished)", flush=True)
      continue

    t0 = time.time()
    print(f"{head}  starting…", flush=True)
    log_path = log_dir / f"{run_name}.log"
    with open(log_path, "w") as log:
      log.write(" ".join(cmd) + "\n\n")
      log.flush()
      rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)

    mins = (time.time() - t0) / 60
    if rc == 0:
      elapsed = (time.time() - t_start) / 60
      left = (len(runs) - i) * mins
      print(
        f"{head}  done in {mins:.1f} min "
        f"(elapsed {elapsed / 60:.1f} h, ~{left / 60:.1f} h left)",
        flush=True,
      )
    else:
      failed.append(run_name)
      print(f"{head}  FAILED rc={rc} after {mins:.1f} min -> {log_path}", flush=True)

  if failed:
    print(f"\n{len(failed)} failed: {', '.join(failed)}")
    return 1
  print(f"\nall done in {(time.time() - t_start) / 3600:.1f} h")
  return 0


if __name__ == "__main__":
  sys.exit(main())

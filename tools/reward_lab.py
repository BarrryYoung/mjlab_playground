"""Reward Lab -- reward sliders and training controls inside mjlab's play viewer.

Runs mjlab's own ``play`` viewer, with an extra panel bolted onto its GUI, so
the robot and the controls live in one window. Each slider does two things:

  * it mutates the *live* reward term, so the viewer's reward bars respond
    immediately (the robot's behaviour does not change -- the policy is fixed);
  * it becomes a CLI override when you hit Start training, which spawns a plain
    ``uv run train ...`` in the background.

The viewer's built-in checkpoint Sync / Use Latest buttons then follow that run
as it trains.

Nothing in site-packages is edited: we subclass ViserPlayViewer and rebind the
name that ``play`` looks up, inside our own process.

    uv run python tools/reward_lab.py Playground-Velocity-Flat-Unitree-Go1 \
        --checkpoint-file logs/rsl_rl/<exp>/<run>/model_1000.pt --viewer viser
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import viser
from mjlab.tasks.registry import load_rl_cfg
from mjlab.viewer.viser.viewer import ViserPlayViewer, format_time_ago

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "logs" / "rsl_rl"

# rsl_rl prints a centred, ANSI-bold "Learning iteration <it>/<total>" banner
# once per iteration; that is a finer-grained progress signal than counting
# checkpoints, which only land every `save_interval` steps.
_ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")


def _dash(name: str) -> str:
  """tyro renders underscores in field names as dashes on the command line."""
  return name.replace("_", "-")


def _is_tunable(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _slider_range(default: float) -> tuple[float, float, float]:
  """Bracket a default value with a range that keeps its sign meaningful."""
  if default == 0.0:
    lo, hi = -1.0, 1.0
  elif default > 0.0:
    lo, hi = 0.0, default * 4.0
  else:
    lo, hi = default * 4.0, 0.0
  return lo, hi, (hi - lo) / 200.0


class RewardLabViewer(ViserPlayViewer):
  """The stock play viewer plus a reward-tuning and training panel."""

  task_id: str = ""

  def setup(self) -> None:
    # The stock setup() keeps its tab group in a local, so anything we add
    # afterwards lands in the root container and shows up under every tab.
    # Intercept the constructor call to get a handle on it.
    gui = self._server.gui
    stock_add_tab_group = gui.add_tab_group
    captured: list[object] = []

    def capture(*args, **kwargs):
      handle = stock_add_tab_group(*args, **kwargs)
      captured.append(handle)
      return handle

    gui.add_tab_group = capture  # type: ignore[method-assign]
    try:
      super().setup()
    finally:
      gui.add_tab_group = stock_add_tab_group  # type: ignore[method-assign]

    self._tabs = captured[0] if captured else gui.add_tab_group()

    self._train: subprocess.Popen | None = None
    self._run_dir: Path | None = None
    self._stock_load = None
    self._last_line = ""
    self._iter = 0
    self._total_iters = 0
    self._overrides: list[tuple[str, object, float, object, str]] = []

    self._experiment = load_rl_cfg(self.task_id).experiment_name
    self._build_panel()
    threading.Thread(target=self._poll_status, daemon=True).start()

  # ── gui ─────────────────────────────────────────────────────────

  def _build_panel(self) -> None:
    gui = self._server.gui
    rewards = self.env.unwrapped.cfg.rewards

    with self._tabs.add_tab("Reward Lab", icon=viser.Icon.FLASK):
      gui.add_markdown(
        "Sliders retune the **live** reward (bars react, gait does not) "
        "and seed the next training run."
      )

      for term_name in sorted(rewards):
        term = rewards[term_name]
        with gui.add_folder(term_name):
          self._add_slider(term_name, term, "weight", float(term.weight))
          for param, value in sorted(term.params.items()):
            if _is_tunable(value):
              self._add_slider(term_name, term, param, float(value))

      with gui.add_folder("View run"):
        gui.add_markdown(
          "Which run the checkpoint dropdown above reads from. "
          "Starting a run switches to it automatically."
        )
        self._run_picker = gui.add_dropdown("run", self._list_runs() or ["—"])
        self._rescan_btn = gui.add_button("Rescan runs")

        @self._run_picker.on_update
        def _(_) -> None:
          name = self._run_picker.value
          if name and name != "—":
            self._repoint_checkpoints(LOG_ROOT / self._experiment / name)

        self._rescan_btn.on_click(lambda _: self._refresh_runs())

      with gui.add_folder("Training"):
        self._run_name = gui.add_text("run name", initial_value="lab")
        self._num_envs = gui.add_number("num envs", 4096, min=64, max=32768, step=64)
        self._iters = gui.add_number("iterations", 3000, min=50, max=50000, step=50)
        self._save_every = gui.add_number("save every", 50, min=10, max=500, step=10)

        self._start_btn = gui.add_button("▶ Start training")
        self._stop_btn = gui.add_button("■ Stop", disabled=True)
        self._reset_btn = gui.add_button("Reset sliders")
        self._progress = gui.add_progress_bar(0.0, color="green")
        self._status = gui.add_html("")

        self._start_btn.on_click(lambda _: self.start_training())
        self._stop_btn.on_click(lambda _: self.stop_training())
        self._reset_btn.on_click(lambda _: self.reset_sliders())

  def _add_slider(self, term_name: str, term, field: str, default: float) -> None:
    """Add a slider that edits `term` in place and records its CLI override."""
    if field == "weight":
      flag = f"--env.rewards.{_dash(term_name)}.weight"
    else:
      flag = f"--env.rewards.{_dash(term_name)}.params.{_dash(field)}"

    lo, hi, step = _slider_range(default)
    handle = self._server.gui.add_slider(field, lo, hi, step, default, hint=flag)

    @handle.on_update
    def _(_) -> None:
      # The reward manager holds a reference to this cfg object and re-reads
      # weight/params every step, so the edit lands on the next frame.
      if field == "weight":
        term.weight = float(handle.value)
      else:
        term.params[field] = float(handle.value)

    self._overrides.append((flag, handle, default, term, field))

  def reset_sliders(self) -> None:
    for _, handle, default, term, field in self._overrides:
      handle.value = default
      if field == "weight":
        term.weight = default
      else:
        term.params[field] = default

  # ── training subprocess ─────────────────────────────────────────

  def _build_command(self) -> list[str]:
    cmd = [
      "uv", "run", "train", self.task_id,
      "--env.scene.num-envs", str(int(self._num_envs.value)),
      "--agent.max-iterations", str(int(self._iters.value)),
      "--agent.save-interval", str(int(self._save_every.value)),
      "--agent.run-name", self._run_name.value,
      "--agent.logger", "tensorboard",
    ]  # fmt: skip
    for flag, handle, _, _, _ in self._overrides:
      cmd += [flag, f"{handle.value:g}"]
    return cmd

  def start_training(self) -> None:
    if self._train is not None and self._train.poll() is None:
      return

    started_at = time.time()
    cmd = self._build_command()
    print("\n$ " + " ".join(cmd) + "\n", flush=True)

    # start_new_session puts the child in its own process group, so Stop can
    # take down uv and the python process it spawns in one shot.
    self._train = subprocess.Popen(
      cmd,
      cwd=REPO,
      start_new_session=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    self._last_line = "starting…"
    self._run_dir = None
    self._iter = 0
    self._total_iters = int(self._iters.value)
    threading.Thread(target=self._pump_output, args=(self._train,), daemon=True).start()
    threading.Thread(target=self._find_run_dir, args=(started_at,), daemon=True).start()

    self._start_btn.disabled = True
    self._stop_btn.disabled = False

  def stop_training(self) -> None:
    proc = self._train
    if proc is not None and proc.poll() is None:
      os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    self._train = None
    self._last_line = ""
    self._start_btn.disabled = False
    self._stop_btn.disabled = True

  def close(self) -> None:
    self.stop_training()
    super().close()

  # ── background bookkeeping ──────────────────────────────────────

  def _pump_output(self, proc: subprocess.Popen) -> None:
    """Mirror the trainer's stdout to our console, keeping the last line."""
    assert proc.stdout is not None
    for line in proc.stdout:
      line = line.rstrip()
      print(line, flush=True)
      match = _ITER_RE.search(line)
      if match:
        self._iter, self._total_iters = int(match.group(1)), int(match.group(2))
      elif line.strip():
        self._last_line = line.strip()[:90]

  def _find_run_dir(self, started_at: float) -> None:
    """Follow the run we just started: find its directory, then select it.

    The directory appears within a second or two, but its first checkpoint only
    lands at `save_interval`, and the picker only lists runs that have one --
    so these are two separate waits.
    """
    exp_dir = LOG_ROOT / self._experiment
    while self._alive():
      if exp_dir.is_dir():
        fresh = [
          d for d in exp_dir.iterdir() if d.is_dir() and d.stat().st_mtime >= started_at
        ]
        if fresh:
          self._run_dir = max(fresh, key=lambda d: d.stat().st_mtime)
          break
      time.sleep(1.0)

    while self._alive() and self._run_dir is not None:
      if any(self._run_dir.glob("model_*.pt")):
        self._refresh_runs(select=self._run_dir.name)
        return
      time.sleep(2.0)

  def _alive(self) -> bool:
    return self._train is not None and self._train.poll() is None

  def _list_runs(self) -> list[str]:
    """Run directories for this task's experiment, newest first."""
    exp_dir = LOG_ROOT / self._experiment
    if not exp_dir.is_dir():
      return []
    runs = [d for d in exp_dir.iterdir() if d.is_dir() and any(d.glob("model_*.pt"))]
    return [d.name for d in sorted(runs, key=lambda d: d.name, reverse=True)]

  def _refresh_runs(self, select: str | None = None) -> None:
    """Rescan the log root, keeping the current selection where possible."""
    runs = self._list_runs()
    if not runs:
      return
    keep = select or self._run_picker.value
    chosen = keep if keep in runs else runs[0]
    self._run_picker.options = runs
    self._run_picker.value = chosen
    # Assigning .value does not necessarily fire on_update (that fires for
    # client-side edits), so repoint explicitly.
    self._repoint_checkpoints(LOG_ROOT / self._experiment / chosen)

  def _repoint_checkpoints(self, run_dir: Path) -> None:
    """Aim the viewer's Sync button at the run we just started.

    play() captures the checkpoint directory from --checkpoint-file at startup,
    so without this the dropdown keeps listing the old run forever. Both hooks
    are plain attributes on the manager, so swapping them is enough.
    """
    mgr = self._ckpt_mgr
    if mgr is None:
      return

    # Capture the untouched loader exactly once. Re-reading it on every repoint
    # would wrap the previous wrapper, nesting one layer per run switch.
    if self._stock_load is None:
      self._stock_load = mgr.load_checkpoint

    def fetch_available() -> list[tuple[str, str]]:
      now = time.time()
      entries = []
      for f in run_dir.glob("model_*.pt"):
        try:
          step = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
          continue
        ago = format_time_ago(int(now - f.stat().st_mtime))
        entries.append((f.name, ago, step))
      entries.sort(key=lambda e: e[2])
      return [(name, ago) for name, ago, _ in entries]

    def load(name: str) -> object:
      # The stock loader joins the name onto the directory it captured at
      # startup; handing it an absolute path makes that join resolve to ours.
      path = run_dir / name
      if not path.exists():
        raise FileNotFoundError(f"{name} is not in {run_dir.name}")
      return self._stock_load(str(path))

    mgr.fetch_available = fetch_available
    mgr.load_checkpoint = load
    mgr.run_name = run_dir.name

    # current_name must name a file that exists in the new run. The stock Sync
    # handler looks it up in the refreshed dropdown options and assigns the
    # result; a stale name from the previous run is not in that list, so the
    # assignment throws and leaves the dropdown's update event cleared --
    # which silently kills checkpoint selection until "Use Latest" repairs it.
    entries = fetch_available()
    if entries:
      mgr.current_name = entries[0][0]
      # Load the newest one on the viewer's own thread rather than from here.
      self.request_action("FETCH_CHECKPOINT", "latest")

  def _poll_status(self) -> None:
    while True:
      running = self._train is not None and self._train.poll() is None
      ckpt = None
      if self._run_dir is not None:
        ckpts = list(self._run_dir.glob("model_*.pt"))
        if ckpts:
          ckpt = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

      if running:
        state = '<span style="color:#4caf50">● training</span>'
      elif self._train is not None:
        state = '<span style="color:#f44336">■ exited</span>'
      else:
        state = '<span style="color:#999">○ idle</span>'

      total = self._total_iters or int(self._iters.value)
      self._progress.value = min(100.0, 100.0 * self._iter / total) if total else 0.0
      self._progress.animated = running

      run = self._run_dir.name if self._run_dir else "—"
      progress = f"iter {self._iter} / {total}"
      latest = f"latest ckpt: {ckpt.stem.split('_')[1]}" if ckpt else "no checkpoint yet"
      hint = (
        "<br><span style='color:#4caf50'>↑ hit Sync to load it</span>" if ckpt else ""
      )

      self._status.content = (
        '<div style="padding:.4em;font-family:monospace;font-size:.75em;'
        'line-height:1.5;color:#ddd;">'
        f"{state} &nbsp; {progress}<br>run: {run}<br>{latest}{hint}"
        f'<div style="color:#888;margin-top:.4em;word-break:break-all;">'
        f"{self._last_line}</div></div>"
      )
      time.sleep(1.0)


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
    sys.exit(f"usage: {sys.argv[0]} <TASK_ID> [play options]")

  RewardLabViewer.task_id = sys.argv[1]

  # `play` looks the viewer class up on its own module, so rebinding the name
  # there is enough -- site-packages stays untouched.
  import mjlab.scripts.play as play_script

  play_script.ViserPlayViewer = RewardLabViewer  # type: ignore[misc]
  if "--viewer" not in sys.argv:
    sys.argv += ["--viewer", "viser"]
  play_script.main()


if __name__ == "__main__":
  main()

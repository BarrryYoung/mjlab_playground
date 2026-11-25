"""Warm-start a checkpoint for a task whose actor / critic inputs grew.

Two kinds of growth, per observation group, in any combination:
  - terms appended at the end (e.g. the critic's height scan);
  - history added to existing terms (e.g. the actor at 6 frames).
The first layer's weights for the old inputs are moved to where those inputs
now sit (for a term with history: its newest frame), every new column is zero.
At step 0 each network computes exactly what it did before; the new columns get
gradient (dL/dW = delta * x) and grow from there. Adam's first moment is zero
on the new columns; the second moment is borrowed (older history frames: from
the newest frame of the same term; appended terms: the row mean), see below.
The rest of the checkpoint (iteration, env step counter, lr, std) is kept, so a
resume continues it.

mjlab flattens history per term, oldest frame first:
  [t1(k-H+1) .. t1(k), t2(k-H+1) .. t2(k), ...]

usage: python tools/warmstart_ckpt.py SRC_CKPT --task T [--out PATH]
then:  train.py T --agent.resume True --agent.load-run <dir of PATH> --agent.load-checkpoint <name>
"""

import argparse
from pathlib import Path

import mjlab_playground  # noqa: F401  (registers the tasks)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--task", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()

src = Path(args.src)
ck = torch.load(src, map_location="cpu", weights_only=False)

cfg = load_env_cfg(args.task)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
om = env.observation_manager
layout = {}
for g in ("actor", "critic"):
  names = om.active_terms[g]
  dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[g]]
  hist = [max(1, c.history_length) for c in om._group_obs_term_cfgs[g]]
  layout[g] = list(zip(names, dims, hist, strict=True))
env.close()


def column_map(g: str, old_width: int) -> tuple[torch.Tensor, torch.Tensor, int, list]:
  """new column of each old column; for every new column, the old column whose
  Adam second moment it borrows (-1: none, use the row mean); new width; notes."""
  new_cols, borrow, notes = [], [], []
  # Old inputs that already are the first terms at their full (history) width:
  # keep them where they are, only the appended terms are new (11-24, e.g. the
  # rough 342 actor + wheel_vel_pos). Single-frame terms map the same either way.
  prefix = 0
  for i, (name, dim, h) in enumerate(layout[g]):
    prefix += dim
    if prefix == old_width and any(hh > 1 for _, _, hh in layout[g][: i + 1]):
      cols = list(range(old_width))
      borrow = cols + [-1] * sum(d for _, d, _ in layout[g][i + 1 :])
      notes = [f"+{n}({d})" for n, d, _ in layout[g][i + 1 :]] + ["old kept as is"]
      return torch.tensor(cols), torch.tensor(borrow), len(borrow), notes
  old_off = new_off = 0
  for name, dim, h in layout[g]:
    per_frame = dim // h
    if old_off < old_width:
      # Old term: one frame wide; it is the newest frame now, and the older
      # frames of the same term borrow its second moment.
      newest = new_off + (h - 1) * per_frame
      new_cols += range(newest, newest + per_frame)
      borrow += list(range(old_off, old_off + per_frame)) * h
      old_off += per_frame
      if h > 1:
        notes.append(f"{name} x{h}")
    else:
      borrow += [-1] * dim
      notes.append(f"+{name}({dim})")
    new_off += dim
  assert old_off == old_width, (
    f"{g}: old width {old_width} does not end on a term boundary"
  )
  return torch.tensor(new_cols), torch.tensor(borrow), new_off, notes


opt_state = ck["optimizer_state_dict"]["state"]
infos = {}
for g in ("actor", "critic"):
  w = ck[f"{g}_state_dict"]["mlp.0.weight"]
  cols, borrow, width, notes = column_map(g, w.shape[1])
  print(f"{g:6s} {w.shape[1]} -> {width}  {', '.join(notes) or 'unchanged'}")
  if width == w.shape[1]:
    continue

  def remap(x: torch.Tensor, width=width, cols=cols) -> torch.Tensor:
    y = x.new_zeros(x.shape[0], width)
    y[:, cols] = x
    return y

  ck[f"{g}_state_dict"]["mlp.0.weight"] = remap(w)
  hits = [s for s in opt_state.values() if s["exp_avg"].shape == w.shape]
  assert len(hits) == 1, (
    f"{g}: {len(hits)} optimizer entries shaped like {tuple(w.shape)}"
  )
  # First moment: zero on the new columns (no gradient history). Second moment:
  # borrowed, not zero. Adam's step counter is global and already in the
  # thousands, so there is no bias correction left: a zero second moment makes
  # the first update of a new column ~3 lr * sign(g), and 285 columns moving
  # like that at once wrecked the warm-started actor (value loss 0.01 -> 0.7,
  # success 99% -> 17% within 5 iterations).
  hits[0]["exp_avg"] = remap(hits[0]["exp_avg"])
  v = hits[0]["exp_avg_sq"]
  v_new = torch.where(borrow >= 0, borrow, 0)
  v_new = v[:, v_new]
  v_new[:, borrow < 0] = v.mean(dim=1, keepdim=True)
  hits[0]["exp_avg_sq"] = v_new
  infos[g] = notes

ck["infos"] = {
  **(ck.get("infos") or {}),
  "warmstart_from": str(src.resolve()),
  "warmstart": infos,
}
out = Path(args.out) if args.out else src.with_name(src.stem + "_ws.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, out)
print(f"wrote {out}")

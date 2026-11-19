"""Warm-start a SkipRNNModel actor (see mjlab_playground.rl.skip_rnn) from an
MLP checkpoint.

Builds the task's runner once to get the new model's parameters (GRU weights at
their fresh init, in the optimizer's parameter order), then:
  - actor MLP: copied; its first layer gets zero columns for the GRU state, so
    the actor computes exactly the old MLP's output at step 0;
  - actor GRU: fresh init;
  - critic: copied, zero columns for terms appended after the old inputs;
  - Adam: the old state for every copied parameter (first moment zero on new
    columns, second moment borrowed from the row mean); GRU params get the
    row-mean second moment of the actor's first layer and a zero first moment.
    Adam's step counter is global and already in the thousands, so a zero
    second moment would make the first update of a new weight ~3 lr * sign(g).

usage: python tools/warmstart_gru.py SRC_CKPT --task T --out PATH
"""

import argparse
import contextlib
import io
from dataclasses import asdict
from pathlib import Path

import mjlab_playground  # noqa: F401  (registers the tasks)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--task", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

old = torch.load(args.src, map_location="cpu", weights_only=False)

env_cfg = load_env_cfg(args.task)
env_cfg.scene.num_envs = 2
with contextlib.redirect_stdout(io.StringIO()):
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda"))
  runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), None, "cuda")
alg = runner.alg
params = list(alg.actor.named_parameters(prefix="actor")) + list(
  alg.critic.named_parameters(prefix="critic")
)
new_sd = {"actor": alg.actor.state_dict(), "critic": alg.critic.state_dict()}
opt_new = alg.optimizer.state_dict()
env.close()

# Old optimizer state is keyed by position in chain(actor.parameters(), critic.parameters()).
old_params = [("actor", k) for k in old["actor_state_dict"]] + [
  ("critic", k) for k in old["critic_state_dict"]
]
# state_dict order == parameters() order here (no buffers in these models).
old_state = old["optimizer_state_dict"]["state"]
old_by_name = {
  f"{g}.{k}": old_state[i] for i, (g, k) in enumerate(old_params) if i in old_state
}

state = {}
for i, (name, p) in enumerate(params):
  g, k = name.split(".", 1)
  ref = old[f"{g}_state_dict"].get(k)
  st = old_by_name.get(name)
  if ref is None:
    # New parameter (the GRU): keep its fresh init; borrow a second moment.
    lay = old_by_name["actor.mlp.0.weight"]["exp_avg_sq"].mean()
    state[i] = {
      "step": next(iter(old_state.values()))["step"].clone(),
      "exp_avg": torch.zeros_like(p, device="cpu"),
      "exp_avg_sq": torch.full_like(p, float(lay), device="cpu"),
    }
    print(f"  new    {name:32s} {tuple(p.shape)}")
    continue
  if ref.shape == p.shape:
    new_sd[g][k] = ref.clone()
    state[i] = {kk: v.clone() for kk, v in st.items()}
    continue
  # Grown input layer: old columns first, new columns appended.
  assert ref.dim() == 2 and ref.shape[0] == p.shape[0] and ref.shape[1] < p.shape[1], (
    name
  )
  n_old, n_new = ref.shape[1], p.shape[1]
  w = torch.zeros(p.shape)
  w[:, :n_old] = ref
  new_sd[g][k] = w
  m = torch.zeros(p.shape)
  m[:, :n_old] = st["exp_avg"]
  v = st["exp_avg_sq"].mean(dim=1, keepdim=True).expand(-1, n_new).clone()
  v[:, :n_old] = st["exp_avg_sq"]
  state[i] = {"step": st["step"].clone(), "exp_avg": m, "exp_avg_sq": v}
  print(f"  grown  {name:32s} {tuple(ref.shape)} -> {tuple(p.shape)}")

opt_new["state"] = state
opt_new["param_groups"][0]["lr"] = old["optimizer_state_dict"]["param_groups"][0]["lr"]
out = {
  "actor_state_dict": new_sd["actor"],
  "critic_state_dict": new_sd["critic"],
  "optimizer_state_dict": opt_new,
  "iter": old["iter"],
  "infos": {
    **(old.get("infos") or {}),
    "warmstart_from": str(Path(args.src).resolve()),
  },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, args.out)
print(f"wrote {args.out}")

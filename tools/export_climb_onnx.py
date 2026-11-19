"""Export one checkpoint of a Go2W Climb policy to ONNX, plus its interface.

The training runner exports an ONNX on every save, overwriting one file per run;
this exports a chosen checkpoint and writes next to it a JSON describing what
the deploy side has to feed and do:

  - actor observation terms in order, with per-term width and scale, and the
    history layout (Blind tasks: 6 frames, per term oldest -> newest,
    flattened, then the terms concatenated; reset back-fills with the first
    frame);
  - joint order (SDK: FR, FL, RR, RL legs, then wheels), default joint
    positions, leg / wheel action scales, kp / kd;
  - the climb-mode bit (1 = climb mode; held by the operator in front of a
    ledge or bar), and the command layout (vx, vy, wz; in climb mode the
    policy was trained on vx 0.3-0.8, vy 0, wz from a heading controller).

The ONNX output is checked against the torch actor on random inputs.

usage: python tools/export_climb_onnx.py <task> <run_glob>@<iter> <out_dir>
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import mjlab_playground  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task, spec, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
g, it = spec.split("@")
ck = sorted(Path("logs/rsl_rl").glob(f"*/{g}"))[-1] / f"model_{it}.pt"
sd = torch.load(ck, map_location="cpu", weights_only=False)["actor_state_dict"]
assert not any("normalizer" in k for k in sd), (
  "actor has an obs normalizer: not handled"
)
idx = sorted(
  {int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith("weight")}
)
layers = []
for n, i in enumerate(idx):
  w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
  lin = torch.nn.Linear(w.shape[1], w.shape[0])
  lin.weight.data.copy_(w)
  lin.bias.data.copy_(b)
  layers.append(lin)
  if n < len(idx) - 1:
    layers.append(torch.nn.ELU())
actor = torch.nn.Sequential(*layers).eval()
obs_dim = layers[0].in_features

onnx_path = out_dir / "policy.onnx"
torch.onnx.export(
  actor,
  torch.zeros(1, obs_dim),
  str(onnx_path),
  input_names=["obs"],
  output_names=["actions"],
  dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
  opset_version=17,
)
from onnx.reference import ReferenceEvaluator  # noqa: E402

ref = ReferenceEvaluator(str(onnx_path))
x = np.random.default_rng(0).normal(size=(64, obs_dim)).astype(np.float32)
with torch.no_grad():
  y_t = actor(torch.from_numpy(x)).numpy()
y_o = ref.run(None, {"obs": x})[0]

# go2w_ctrl (C++ onnxruntime) sizes its input tensor from the model's input
# shape, so a symbolic batch dim becomes -1 and it aborts: pin batch to 1 and
# write a single file (no .onnx.data).
import onnx  # noqa: E402

m = onnx.load(str(onnx_path))
for t in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
  for d in t.type.tensor_type.shape.dim:
    if d.dim_param:
      d.ClearField("dim_param")
      d.dim_value = 1
onnx.checker.check_model(m)
for f in out_dir.glob("policy.onnx*"):
  f.unlink()
onnx.save_model(m, str(onnx_path), save_as_external_data=False)
y_1 = ReferenceEvaluator(str(onnx_path)).run(None, {"obs": x[:1]})[0]
assert np.abs(y_1 - y_t[:1]).max() < 1e-4
err = float(np.abs(y_t - y_o).max())

# Interface, read off the task's play env.
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 1
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device="cpu")
om = env.observation_manager
group = cfg.observations["actor"]
terms = []
for name, tcfg in zip(
  om.active_terms["actor"], om._group_obs_term_cfgs["actor"], strict=True
):
  dim = int(tcfg.func(env, **tcfg.params).shape[-1])
  terms.append(
    {
      "name": name,
      "dim": dim,
      "scale": tcfg.scale if tcfg.scale is not None else 1.0,
      "clip": list(tcfg.clip) if tcfg.clip is not None else None,
    }
  )
hist = group.history_length or 1
assert sum(t["dim"] for t in terms) * hist == obs_dim, (terms, hist, obs_dim)
robot = env.scene["robot"]
jp = env.action_manager.get_term("joint_pos")
wv = env.action_manager.get_term("wheel_vel")
names = list(robot.joint_names)
default = robot.data.default_joint_pos[0].tolist()
iface = {
  "task": task,
  "checkpoint": str(ck),
  "obs_dim": obs_dim,
  "history_length": hist,
  "history_layout": "per term: oldest -> newest frames, flattened; terms concatenated in the order below; "
  "after a reset every frame of the buffer is the first observation",
  "terms": terms,
  "joint_order_sdk": list(jp._target_names),
  "wheel_order_sdk": list(wv._target_names),
  "default_joint_pos": dict(zip(names, default, strict=True)),
  "leg_action_scale": float(
    jp._scale if not torch.is_tensor(jp._scale) else jp._scale.flatten()[0]
  ),
  "wheel_action_scale": float(
    wv._scale if not torch.is_tensor(wv._scale) else wv._scale.flatten()[0]
  ),
  "leg_target": "default_joint_pos + leg_action_scale * action[:12] (SDK order)",
  "wheel_target": "wheel velocity = wheel_action_scale * action[12:] (rad/s)",
  "kp_kd_legs": [25.0, 0.5],
  "climb_mode": "last per-frame term 'climb_mode' (Blind tasks): 1.0 in climb mode, else 0.0",
  "command": "[vx, vy, wz] in the body frame; climb mode was trained on vx 0.3-0.8, vy 0, "
  "wz = clip(0.5 * heading error, -1, 1); command deadzone 0.05 per axis (FlatPose tasks)",
  "onnx_max_abs_err_vs_torch": err,
}


def _plain(o):
  if torch.is_tensor(o):
    o = o.detach().cpu()
    return o.item() if o.numel() == 1 else o.tolist()
  if isinstance(o, (np.ndarray, np.generic)):
    return o.tolist()
  if isinstance(o, (list, tuple)):
    return [_plain(v) for v in o]
  if isinstance(o, dict):
    return {k: _plain(v) for k, v in o.items()}
  return o


iface = _plain(iface)
(out_dir / "interface.json").write_text(json.dumps(iface, indent=1, ensure_ascii=False))
print(f"{ck} -> {onnx_path}  (obs {obs_dim}, max |onnx - torch| {err:.2e})")
print(
  json.dumps(
    {
      k: iface[k]
      for k in ("history_length", "terms", "leg_action_scale", "wheel_action_scale")
    },
    indent=None,
  )
)

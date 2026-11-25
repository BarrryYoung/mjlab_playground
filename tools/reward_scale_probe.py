"""用现有策略在训练环境里跑一段，统计各奖励项的 Episode_Reward（与训练日志同口径），
用来给新奖励项定权重。新项权重临时设为 -1，输出即“单位权重下的量级”。
usage: python tools/reward_scale_probe.py CKPT --task T [--envs 2048] [--steps 3000] [--term stand_displacement]
"""
import argparse, collections, contextlib, io, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_playground  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--task", required=True)
ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--term", default="stand_displacement")
a = ap.parse_args()
gb.DEVICE = "cuda:0"
cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.envs
cfg.rewards[a.term].weight = -1.0
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
pol = gb._load_policy(Path(a.ckpt))
obs, _ = env.reset()
acc = collections.defaultdict(list)
with torch.no_grad():
  for t in range(a.steps):
    obs, _, _, _, extras = env.step(pol(obs["actor"]))
    for k, v in (extras.get("log") or {}).items():
      if k.startswith("Episode_Reward/") or k.startswith("Curriculum/terrain_levels/mean"):
        acc[k].append(float(v))
print(f"{a.ckpt}  task={a.task}  steps={a.steps}")
for k, v in sorted(acc.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1]))):
  m = sum(v) / len(v)
  if abs(m) > 1e-5:
    print(f"  {k:48s} {m:+.4f}")

"""Time env.step of a rough task under different sim settings, with a policy driving.

For each setting: build the env, run the policy (no grad) for --warm steps, then
time --steps steps. Also records the per-world constraint count (nefc) and the
total contact count (nacon) peaks, to size njmax / nconmax.

usage: python tools/rough_speed_bench.py --ckpt PATH [--envs 4096] [--steps 200]
"""

import argparse
import contextlib
import io
import time

import mjlab_playground  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Playground-Velocity-Rough-Heading-H6-Unitree-Go2W")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--envs", type=int, default=4096)
ap.add_argument("--warm", type=int, default=100)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--only", default=None, help="comma list of setting names")
args = ap.parse_args()

sd = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)[
  "actor_state_dict"
]
idx = sorted(
  {int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith("weight")}
)
layers = []
for n, i in enumerate(idx):
  w = sd[f"mlp.{i}.weight"]
  lin = torch.nn.Linear(w.shape[1], w.shape[0]).cuda()
  lin.weight.data.copy_(w)
  lin.bias.data.copy_(sd[f"mlp.{i}.bias"])
  layers.append(lin)
  if n < len(idx) - 1:
    layers.append(torch.nn.ELU())
actor = torch.nn.Sequential(*layers).eval()

# name -> (njmax, ccd_iterations, contact_sensor_maxmatch, terrain scan on)
SETTINGS = {
  "current": (1500, 100, 500, True),
  "njmax600": (600, 100, 500, True),
  "njmax400": (400, 100, 500, True),
  "ccd50": (1500, 50, 500, True),
  "maxmatch64": (1500, 100, 64, True),
  "lean": (400, 50, 64, True),
}
names = args.only.split(",") if args.only else list(SETTINGS)

for name in names:
  njmax, ccd, maxmatch, _ = SETTINGS[name]
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.envs
  cfg.sim.njmax = njmax
  cfg.sim.mujoco.ccd_iterations = ccd
  cfg.sim.contact_sensor_maxmatch = maxmatch
  with contextlib.redirect_stdout(io.StringIO()):
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
  # Spread envs over all rows so stairs at every height are in the sample.
  t = env.scene.terrain
  t.terrain_levels[:] = torch.randint_like(t.terrain_levels, t.max_terrain_level)
  t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
  obs, _ = env.reset()
  d = env.sim.wp_data
  nefc_max, nacon_max, overflow = 0, 0, 0
  with torch.no_grad():
    # Warm-up doubles as the count pass (reading counts syncs, so not timed).
    for _ in range(args.warm):
      obs, *_ = env.step(actor(obs["actor"]))
      nefc = torch.as_tensor(d.nefc.numpy())
      nefc_max = max(nefc_max, int(nefc.max()))
      overflow += int((nefc >= njmax).sum())
      nacon_max = max(nacon_max, int(d.nacon.numpy()[0]))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
      obs, *_ = env.step(actor(obs["actor"]))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
  sps = args.steps * args.envs / dt
  print(
    f"{name:11s} njmax {njmax:4d} ccd {ccd:3d} maxmatch {maxmatch:3d} | "
    f"{sps / 1e3:6.1f} k steps/s | peak nefc/world {nefc_max:4d} (at cap: {overflow}) | "
    f"peak nacon {nacon_max} ({nacon_max / args.envs:.1f}/world)",
    flush=True,
  )
  env.close()
  del env
  torch.cuda.empty_cache()

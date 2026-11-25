"""0 指令站立时，策略输出的目标轮速分布 vs 实际轮速。usage: python tools/wheel_cmd_probe.py CKPT --task T"""
import argparse, contextlib, io, sys, types
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
import mjlab_playground  # noqa
from mjlab_playground.velocity import mdp
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--task", required=True); a = ap.parse_args()
D = "cuda:0"; gb.DEVICE = D
cfg = load_env_cfg(a.task, play=True); cfg.scene.num_envs = 128
cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
cfg.episode_length_s = 1e6; cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None); cfg.events.pop("init_rough_terrain_at_top", None)
cfg.observations["actor"].enable_corruption = True
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
term = env.command_manager.get_term("twist")
def fixed(self, ids):
  self.vel_command_b[ids] = 0
  for n in dir(self):
    v = getattr(self, n, None)
    if n.startswith("is_") and isinstance(v, torch.Tensor) and v.dtype == torch.bool: v[ids] = False
term._resample_command = types.MethodType(fixed, term); term._update_command = types.MethodType(lambda s: s.vel_command_b.zero_(), term)
pol = gb._load_policy(Path(a.ckpt)); obs, _ = env.reset(); r = env.scene["robot"]
wid = r.find_joints(["FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"], preserve_order=True)[0]
tg, ac, mean_tg = [], [], []
with torch.no_grad():
  for k in range(1000):
    act = pol(obs["actor"]); obs, *_ = env.step(act)
    if k >= 250:
      t = act[:, 12:16] * 5.0
      tg.append(t.abs().flatten()); ac.append(r.data.joint_vel[:, wid].abs().flatten()); mean_tg.append(t)
tg = torch.cat(tg); ac = torch.cat(ac); mt = torch.stack(mean_tg).mean(0).abs().flatten()
q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=D)
print("|目标轮速| 分位 10/25/50/75/90 (rad/s):", [round(x, 2) for x in torch.quantile(tg[:200000], q).tolist()])
print("|实际轮速| 分位 10/25/50/75/90 (rad/s):", [round(x, 2) for x in torch.quantile(ac[:200000], q).tolist()])
print("每个轮子 15 s 平均目标轮速(带符号)的绝对值 分位:", [round(x, 3) for x in torch.quantile(mt, q).tolist()])
for th in (0.5, 1.0, 2.0, 3.0): print(f"  |目标| < {th}: {(tg < th).float().mean() * 100:.0f}%")

"""wheelpos 热启动第二步：把 joint_vel 最新帧 4 个轮子列的权重对半分给 wheel_vel_pos 4 列。
先用 tools/warmstart_ckpt.py 扩列（新列为 0），再跑本脚本。初始时网络输出不变（两路输入稳态相等）。
usage: python tools/split_wheelpos_init.py IN_CKPT OUT_CKPT --task T
"""
import argparse
import mjlab_playground  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--task", required=True)
a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False)
cfg = load_env_cfg(a.task); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
om = env.observation_manager
names = om.active_terms["actor"]; dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["actor"]]
off = dict(zip(names, [sum(dims[:i]) for i in range(len(dims))]))
jv = om._group_obs_term_cfgs["actor"][names.index("joint_vel")]
hist = jv.history_length
jn = list(env.scene["robot"].joint_names) if not jv.params["asset_cfg"].joint_names or jv.params["asset_cfg"].joint_names == ".*" else jv.params["asset_cfg"].joint_names
env.close()
per = dims[names.index("joint_vel")] // hist
wheels = [jn.index(n) for n in ("FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint")]
old = [off["joint_vel"] + (hist - 1) * per + i for i in wheels]
new = list(range(off["wheel_vel_pos"], off["wheel_vel_pos"] + 4))
print("joint_vel per-frame", per, "wheel idx", wheels, "old cols", old, "new cols", new)
W0 = ck["actor_state_dict"]["mlp.0.weight"]
assert W0.shape[1] == new[-1] + 1 and (W0[:, new] == 0).all(), W0.shape
W = W0.clone(); half = W[:, old] / 2; W[:, old] = half; W[:, new] = half
ck["actor_state_dict"]["mlp.0.weight"] = W
hits = [s for s in ck["optimizer_state_dict"]["state"].values() if s["exp_avg"].shape == W.shape]
assert len(hits) == 1
hits[0]["exp_avg_sq"][:, new] = hits[0]["exp_avg_sq"][:, old]
ck.setdefault("infos", {})
if ck["infos"] is None: ck["infos"] = {}
ck["infos"]["split_init"] = {"old_cols": old, "new_cols": new}
torch.save(ck, a.out)
print("saved", a.out, "new col norms", W[:, new].norm(dim=0).tolist(), "old", W[:, old].norm(dim=0).tolist())

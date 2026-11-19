"""Does wheel friction limit stair climbing? Pin the wheel friction per env and measure.

Envs are laid out over (column, row, friction). Each runs one episode at a fixed
forward command from the tile origin (yaw +-5 deg), no pushes, no noise. Reported
per group: distance along +x, height change dz (tells which direction the
column really climbs), fall rate, and wheel slip: the tangential speed of the
wheel's material point at the contact, v_center + w x (p_contact - center),
minus its normal component (0 when rolling, large when spinning or sliding).

usage: python tools/friction_bench.py --ckpt RUN_DIR [--iter N]
         [--cols stairs_up,stairs_down,hf_pyramid_slope] [--rows 3,5,7,9]
         [--mu 0.3,0.5,0.7,1.0] [--e 16] [--vx 0.6] [--t 10]
"""

import argparse
import contextlib
import io
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.dr import geom_friction  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab_playground.asset_zoo.robots.unitree_go2w.go2w_constants import (  # noqa: E402
  WHEEL_GEOM_NAMES,
)
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Playground-Velocity-Rough-Straight-H6-Unitree-Go2W")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--iter", default=None)
ap.add_argument("--cols", default="stairs_up,stairs_down,hf_pyramid_slope")
ap.add_argument("--rows", default="3,5,7,9")
ap.add_argument("--mu", default="0.3,0.5,0.7,1.0")
ap.add_argument("--e", type=int, default=16)
ap.add_argument("--vx", type=float, default=0.6)
ap.add_argument("--t", type=float, default=10.0)
args = ap.parse_args()

if args.iter:
  ck = Path(args.ckpt) / f"model_{args.iter}.pt"
else:
  ck = sorted(
    Path(args.ckpt).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
  )[-1]
print(f"checkpoint {ck}")
policy = gb._load_policy(ck)

cfg = load_env_cfg(args.task, play=True)
gen = cfg.scene.terrain.terrain_generator
all_cols = list(gen.sub_terrains)
cols = args.cols.split(",")
rows = [int(x) for x in args.rows.split(",")]
mus = [float(x) for x in args.mu.split(",")]
E = args.e
G = [(c, r, m) for c in cols for r in rows for m in mus]
cfg.scene.num_envs = len(G) * E
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.events.pop("foot_friction", None)
cfg.observations["actor"].enable_corruption = False
tw = cfg.commands["twist"]
cfg.commands["twist"] = replace(tw, resampling_time_range=(1e6, 1e6))
cfg.scene.sensors = (cfg.scene.sensors or ()) + (
  ContactSensorCfg(
    name="wheel_contact_bench",
    primary=ContactMatch(mode="geom", pattern=WHEEL_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "pos", "normal"),
    reduce="maxforce",
    num_slots=1,
  ),
)
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)

N = env.num_envs
terrain = env.scene.terrain
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
sensor = env.scene["wheel_contact_bench"]
gid = torch.tensor([g for g in range(len(G)) for _ in range(E)], device=D)
col_t = torch.tensor([all_cols.index(G[g][0]) for g in gid.tolist()], device=D)
row_t = torch.tensor([G[g][1] for g in gid.tolist()], device=D)
mu_t = torch.tensor([G[g][2] for g in gid.tolist()], device=D)

wheel_cfg = SceneEntityCfg("robot", geom_names=WHEEL_GEOM_NAMES)
wheel_cfg.resolve(env.scene)
for m in mus:
  ids = torch.nonzero(mu_t == m).squeeze(1)
  geom_friction(env, ids, ranges=(m, m), asset_cfg=wheel_cfg)
wheel_body_ids, _ = robot.find_bodies(
  [n.replace("_collision", "_link") for n in WHEEL_GEOM_NAMES], preserve_order=True
)

env.reset()
terrain.terrain_levels[:] = row_t
terrain.terrain_types[:] = col_t
terrain.env_origins[:] = terrain.terrain_origins[row_t, col_t]
st = robot.data.default_root_state.clone()
st[:, :3] = terrain.env_origins + torch.tensor([0.0, 0.0, 0.42], device=D)
yaw = (torch.rand(N, device=D) * 2 - 1) * math.radians(5)
st[:, 3:7] = torch.stack(
  [
    torch.cos(yaw / 2),
    torch.zeros_like(yaw),
    torch.zeros_like(yaw),
    torch.sin(yaw / 2),
  ],
  1,
)
st[:, 7:] = 0.0
robot.write_root_state_to_sim(st)
robot.write_joint_state_to_sim(
  robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
)
cmd = torch.zeros_like(term.vel_command_b)
cmd[:, 0] = args.vx


def set_cmd():
  term.vel_command_b[:] = cmd
  if hasattr(term, "is_standing_env"):
    term.is_standing_env[:] = False


set_cmd()
obs, *_ = env.step(torch.zeros(N, env.action_manager.total_action_dim, device=D))
p0 = robot.data.root_link_pos_w.clone()
fell = torch.zeros(N, dtype=torch.bool, device=D)
slip_sum = torch.zeros(N, device=D)
slip_big = torch.zeros(N, device=D)
cnt = torch.zeros(N, device=D)
steps = int(args.t / env.step_dt)
with torch.no_grad():
  for _ in range(steps):
    set_cmd()
    obs, *_ = env.step(policy(obs["actor"]))
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    fell |= tilt > math.radians(80)
    found = (
      sensor.data.found.squeeze(-1) > 0
      if sensor.data.found.dim() == 3
      else sensor.data.found > 0
    )
    pos = sensor.data.pos.reshape(N, 4, 3)
    nrm = sensor.data.normal.reshape(N, 4, 3)
    c = robot.data.body_link_pos_w[:, wheel_body_ids]
    v = robot.data.body_link_lin_vel_w[:, wheel_body_ids]
    w = robot.data.body_link_ang_vel_w[:, wheel_body_ids]
    vp = v + torch.cross(w, pos - c, dim=-1)
    vt = vp - (vp * nrm).sum(-1, keepdim=True) * nrm
    s = vt.norm(dim=-1)  # [N, 4]
    alive = (~fell).float()[:, None]
    f = found.float() * alive
    slip_sum += (s * f).sum(1)
    slip_big += ((s > 0.1).float() * f).sum(1)
    cnt += f.sum(1)
d = robot.data.root_link_pos_w - p0

H = gen.sub_terrains
print(
  f"\nvx {args.vx} m/s, {args.t:.0f} s, {E} envs/group. dist = +x progress (m), "
  "slip = mean tangential speed at the wheel contact (m/s), slip>0.1 = share of "
  "contact-steps above 0.1 m/s"
)
print(
  f"\n{'column':18s} {'row':>3s} {'mu':>4s} {'dist':>6s} {'dz':>6s} {'fell%':>6s} "
  f"{'slip':>6s} {'>0.1%':>6s}"
)
for g, (c, r, m) in enumerate(G):
  s = gid == g
  print(
    f"{c:18s} {r:3d} {m:4.1f} {d[s, 0].mean():6.2f} {d[s, 2].mean():+6.2f} "
    f"{fell[s].float().mean() * 100:6.0f} "
    f"{(slip_sum[s].sum() / cnt[s].sum().clamp(min=1)):6.3f} "
    f"{(slip_big[s].sum() / cnt[s].sum().clamp(min=1)) * 100:6.0f}"
  )

"""Obstacle test report for the Go2W Climb policies: long bars and platforms.

One big test field (a single tile named "high_platform", so the climb bit reads
1 everywhere) holds a lane per obstacle spec, side by side:
  bars       h 30/35/40/45/50 cm x width (travel direction) 20/40/60 cm, 4 m long
  platforms  h 30..62 cm (9 heights), 4 m x 4 m boxes
Each env runs one condition on its own lane, deterministic policy, actor
observation noise on, the task's startup randomization (friction, mass, PD,
delay) as in training, no pushes. Conditions:

  bar    speed vx 0.3/0.5/0.8 x approach angle 0-60 deg step 10 x run-up 0.4/1/2 m
  edge   bar end: lateral offset of the base centre from the bar's end
         -0.45/-0.3/-0.15/0/+0.15/+0.3 m (- = over the bar), vx 0.5, square on, run-up 1 m
  up     platform, same speed x angle x run-up grid as bar
  down   start on the platform, drive off its edge, same grid

The angle is held (heading target = start yaw), so the robot meets the face
obliquely, aimed at the middle of the obstacle; run-up = distance from the base
centre to the face along the travel direction (0.4 m square on: front wheels
about 10 cm from the face), but never closer than 0.35 m in x, so at steep
angles the body does not start inside the obstacle.

Outcome per run (first event wins, T seconds):
  success  bar / edge: base > 0.5 m past the far face and back on the floor;
           up: base > 0.5 m onto the platform and on top; down: base > 0.5 m off
           the edge and on the floor
  fall     tilt > 80 deg
  stuck    neither within T
Contact (until the outcome): any trunk / thigh contact with the terrain.

usage: python tools/obstacle_report.py <task> <run_glob>@<iter> <tag>
env:   REPS (16)  T (20)  PHASES (bar,edge,up,down)  NENV (4096)
       ANGS (0,10,...,60)  approach angles, deg
       FRICS (unset)  e.g. 0.3,0.5,1.0: rerun the grid with every env's wheel friction
              fixed to each value (overrides the startup draw; wheels have contact
              priority, so this is the contact friction)
out:   /tmp/obstacle_report/<tag>.csv  (one row per run) + summary on stdout
"""

import contextlib
import csv
import io
import itertools
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gait_bench as gb  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)  # noqa: E402
from mjlab.terrains.utils import make_plane  # noqa: E402
from mjlab_playground.velocity import mdp  # noqa: E402

D = "cuda:0"
gb.DEVICE = D
REPS = int(os.environ.get("REPS", "16"))
T = float(os.environ.get("T", "20"))
PHASES = os.environ.get("PHASES", "bar,edge,up,down").split(",")
NENV = int(os.environ.get("NENV", "4096"))
OUT = Path("/tmp/obstacle_report")

BAR_H = (0.30, 0.35, 0.40, 0.45, 0.50)
BAR_W = (0.20, 0.40, 0.60)
PLAT_H = (0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62)
VX = (0.3, 0.5, 0.8)
ANG = tuple(float(a) for a in os.environ.get("ANGS", "0,10,20,30,40,50,60").split(","))
FRICS = (
  [float(f) for f in os.environ["FRICS"].split(",")]
  if os.environ.get("FRICS")
  else [None]
)
RUN = (0.4, 1.0, 2.0)
EDGE = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30)
LEN = 4.0  # bar length / platform side (m)
PITCH_X, PITCH_Y = 10.0, 6.0  # lane spacing (m)

# Lanes: (kind, h, w). Local x of the near face is the lane's x; obstacle spans +x.
LANES = [("bar", h, w) for h in BAR_H for w in BAR_W] + [
  ("plat", h, LEN) for h in PLAT_H
]
NCOL = 4
NROW = math.ceil(len(LANES) / NCOL)
SIZE = (NCOL * PITCH_X, NROW * PITCH_Y)


def lane_xy(i: int) -> tuple[float, float]:
  """Near-face x and centre-line y of lane i, relative to the tile centre."""
  c, r = i % NCOL, i // NCOL
  return (c - (NCOL - 1) / 2) * PITCH_X - 1.0, (r - (NROW - 1) / 2) * PITCH_Y


@dataclass(kw_only=True)
class ObstacleFieldTerrainCfg(SubTerrainCfg):
  def function(self, difficulty, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    cx, cy = 0.5 * self.size[0], 0.5 * self.size[1]
    geoms = [
      TerrainGeometry(geom=g, color=(0.5, 0.5, 0.5, 1.0))
      for g in make_plane(body, self.size, 0.0, center_zero=False)
    ]
    for i, (kind, h, w) in enumerate(LANES):
      x0, y0 = lane_xy(i)
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(w / 2, LEN / 2, h / 2),
        pos=(cx + x0 + w / 2, cy + y0, h / 2),
      )
      geoms.append(
        TerrainGeometry(
          geom=g,
          color=(0.85, 0.55, 0.20, 1.0) if kind == "bar" else (0.35, 0.55, 0.80, 1.0),
        )
      )
    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


def conditions() -> list[dict]:
  rows = []
  bars = [i for i, lane in enumerate(LANES) if lane[0] == "bar"]
  plats = [i for i, lane in enumerate(LANES) if lane[0] == "plat"]
  grid = list(itertools.product(VX, ANG, RUN))
  if "bar" in PHASES:
    rows += [
      dict(phase="bar", lane=i, vx=v, ang=a, run=r, off=0.0)
      for i in bars
      for v, a, r in grid
    ]
  if "edge" in PHASES:
    rows += [
      dict(phase="edge", lane=i, vx=0.5, ang=0.0, run=1.0, off=o)
      for i in bars
      for o in EDGE
    ]
  for ph in ("up", "down"):
    if ph in PHASES:
      rows += [
        dict(phase=ph, lane=i, vx=v, ang=a, run=r, off=0.0)
        for i in plats
        for v, a, r in grid
      ]
  out = []
  for c in rows:
    kind, h, w = LANES[c["lane"]]
    for rep in range(REPS):
      out.append(dict(c, h=h, w=w, rep=rep))
  return out


task, spec, tag = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_env_cfg(task, play=True)
gen = cfg.scene.terrain.terrain_generator
cfg.scene.terrain.terrain_generator = replace(
  gen,
  size=SIZE,
  num_rows=1,
  num_cols=1,
  curriculum=False,
  sub_terrains={"high_platform": ObstacleFieldTerrainCfg(proportion=1.0)},
)
cfg.scene.terrain.max_init_terrain_level = None
cfg.scene.num_envs = NENV
cfg.episode_length_s = 1e6
cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
cfg.events.pop("push_robot", None)
cfg.curriculum = {}
cfg.observations["actor"].enable_corruption = True
with contextlib.redirect_stdout(io.StringIO()):
  env = ManagerBasedRlEnv(cfg, device=D)
terrain = env.scene.terrain
robot = env.scene["robot"]
term = env.command_manager.get_term("twist")
touch = {k: env.scene[f"{k}_ground_touch"] for k in ("trunk", "thigh")}
origin = terrain.terrain_origins[0, 0].clone()

g, it = spec.split("@")
ck = sorted(Path("logs/rsl_rl").glob(f"*/{g}"))[-1] / f"model_{it}.pt"
policy = gb._load_policy(ck)

wheel_local = torch.tensor(robot.find_geoms(".*_wheel_collision")[0], device=D)
wheel_gid = robot.indexing.geom_ids.to(D)[wheel_local].long()

cmd_vx = torch.zeros(NENV, device=D)
cmd_yaw = torch.zeros(NENV, device=D)


def _fixed(self, ids):
  self.vel_command_b[ids, 0] = cmd_vx[ids]
  self.vel_command_b[ids, 1:] = 0.0
  self.heading_target[ids] = cmd_yaw[ids]
  self.is_heading_env[ids] = True
  self.is_standing_env[ids] = False
  for flag in ("is_world_env", "is_forward_env", "is_low_env"):
    if hasattr(self, flag):
      getattr(self, flag)[ids] = False
  self._climb[ids] = True
  self._axis_pending[ids] = False


term._resample_command = _fixed.__get__(term)


def run_batch(conds: list[dict], seed: int) -> list[dict]:
  gen_ = torch.Generator(device="cpu").manual_seed(seed)
  # Per-env start state, success region, in lane-local coordinates.
  x0 = torch.zeros(NENV)
  y0 = torch.zeros(NENV)
  z0 = torch.zeros(NENV)
  yaw = torch.zeros(NENV)
  vx = torch.zeros(NENV)
  lx = torch.zeros(NENV)
  ly = torch.zeros(NENV)
  hh = torch.zeros(NENV)
  ww = torch.zeros(NENV)
  kind = torch.zeros(NENV, dtype=torch.long)  # 0 bar/edge, 1 up, 2 down, -1 idle
  kind[:] = -1
  jit_y = (torch.rand(NENV, generator=gen_) - 0.5) * 0.10
  jit_a = (torch.rand(NENV, generator=gen_) - 0.5) * math.radians(4.0)
  for e, c in enumerate(conds):
    fx, fy = lane_xy(c["lane"])
    lx[e], ly[e], hh[e], ww[e] = fx, fy, c["h"], c["w"]
    a = math.radians(c["ang"])
    vx[e] = c["vx"]
    if c["phase"] in ("bar", "edge", "up"):
      kind[e] = 0 if c["phase"] != "up" else 1
      d = max(c["run"] * math.cos(a), 0.35)  # x distance to the face
      x0[e] = fx - d
      y0[e] = (
        fy - d * math.tan(a) + (LEN / 2 + c["off"] if c["phase"] == "edge" else 0.0)
      )
      yaw[e] = a
      z0[e] = 0.0
    else:  # down: start on the platform, drive off its near face (-x)
      kind[e] = 2
      d = max(c["run"] * math.cos(a), 0.35)
      x0[e] = fx + d
      y0[e] = fy + d * math.tan(a)
      yaw[e] = math.pi + a
      z0[e] = c["h"]
  y0 += jit_y
  yaw += jit_a
  idle = kind < 0
  x0[idle] = lane_xy(0)[0] - 3.0  # parked on the floor, ignored
  y0[idle] = lane_xy(0)[1]

  to = lambda t: t.to(D)  # noqa: E731
  x0, y0, z0, yaw, vx, lx, ly, hh, ww, kind = map(
    to, (x0, y0, z0, yaw, vx, lx, ly, hh, ww, kind)
  )
  cmd_vx[:] = torch.where(kind >= 0, vx, torch.zeros_like(vx))
  cmd_yaw[:] = yaw

  torch.manual_seed(seed)
  env.reset()
  terrain.env_origins[:] = origin
  st = robot.data.default_root_state.clone()
  st[:, 0] = origin[0] + x0
  st[:, 1] = origin[1] + y0
  st[:, 2] = origin[2] + z0 + 0.42
  z = torch.zeros_like(yaw)
  st[:, 3:7] = torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], 1)
  st[:, 7:] = 0.0
  robot.write_root_state_to_sim(st)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
  )
  term._resample_command(torch.arange(NENV, device=D))
  obs, *_ = env.step(torch.zeros(NENV, env.action_manager.total_action_dim, device=D))

  done = torch.zeros(NENV, dtype=torch.bool, device=D)
  ok = torch.zeros_like(done)
  fell = torch.zeros_like(done)
  t_ok = torch.full((NENV,), float("nan"), device=D)
  hit = {k: torch.zeros_like(done) for k in touch}
  with torch.no_grad():
    for k in range(int(T / env.step_dt)):
      obs, *_ = env.step(policy(obs["actor"]))
      p = robot.data.root_link_pos_w - origin
      px, pz = p[:, 0] - lx, p[:, 2]
      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
      f = (tilt > math.radians(80)) & ~done
      s_bar = (kind == 0) & (px > ww + 0.5) & (pz < 0.55)
      s_up = (kind == 1) & (px > 0.5) & (pz > hh + 0.25)
      s_down = (kind == 2) & (px < -0.5) & (pz < 0.55)
      s = (s_bar | s_up | s_down) & ~done & ~f
      for key, sen in touch.items():
        hit[key] |= (sen.data.found > 0).reshape(NENV, -1).any(1) & ~done
      fell |= f
      ok |= s
      t_ok[s] = (k + 1) * env.step_dt
      done |= f | s
  pf = (robot.data.root_link_pos_w - origin).cpu()
  res = []
  for e, c in enumerate(conds):
    res.append(
      dict(
        c,
        success=int(ok[e]),
        fall=int(fell[e]),
        stuck=int(not done[e]),
        trunk=int(hit["trunk"][e]),
        thigh=int(hit["thigh"][e]),
        t=round(float(t_ok[e]), 2),
        # final base position relative to the lane's near face / centre line
        fx=round(float(pf[e, 0] - lx[e]), 2),
        fy=round(float(pf[e, 1] - ly[e]), 2),
        fz=round(float(pf[e, 2]), 2),
      )
    )
  return res


def pct(rows, key):
  return 100.0 * sum(r[key] for r in rows) / max(len(rows), 1)


def cell(rows):
  return (
    f"{pct(rows, 'success'):3.0f}/{pct(rows, 'fall'):2.0f}/{pct(rows, 'stuck'):2.0f}"
  )


def summarize(rows):
  print(f"\n### {ck.parent.name} @ {it}   {task}")
  print(
    f"    reps {REPS}, T {T}s, obs noise on, startup DR on.  cells: success/fall/stuck %;"
    " contact: trunk/thigh %"
  )
  for ph in ("bar", "edge", "up", "down"):
    R = [r for r in rows if r["phase"] == ph]
    if not R:
      continue
    print(
      f"\n[{ph}]  all: {cell(R)}   contact {pct(R, 'trunk'):.0f}/{pct(R, 'thigh'):.0f}"
    )
    if ph in ("bar", "edge"):
      cols = BAR_W
      print("  h \\ width  " + "".join(f"{w * 100:>16.0f}cm" for w in cols))
      for h in BAR_H:
        cs = []
        for w in cols:
          S = [r for r in R if r["h"] == h and r["w"] == w]
          cs.append(f"{cell(S):>12s} {pct(S, 'trunk'):2.0f}/{pct(S, 'thigh'):2.0f}")
        print(f"  {h * 100:4.0f} cm   " + "  ".join(cs))
    else:
      for h in PLAT_H:
        S = [r for r in R if r["h"] == h]
        print(
          f"  {h * 100:4.0f} cm   {cell(S):>12s}   contact {pct(S, 'trunk'):3.0f}/{pct(S, 'thigh'):3.0f}"
        )
    if ph == "edge":
      print(
        "  by offset (m, - = over the bar): "
        + " | ".join(f"{o:+.2f}: {cell([r for r in R if r['off'] == o])}" for o in EDGE)
      )
    else:
      for key, vals, unit in (
        ("vx", VX, "m/s"),
        ("ang", ANG, "deg"),
        ("run", RUN, "m"),
      ):
        print(
          f"  by {key:3s} ({unit}): "
          + " | ".join(f"{v:g}: {cell([r for r in R if r[key] == v])}" for v in vals)
        )
    # worst cells: condition combos (over reps) with success < 90 %
    key = (
      (lambda r: (r["h"], r["w"], r["off"]))
      if ph == "edge"
      else (lambda r: (r["h"], r["w"], r["vx"], r["ang"], r["run"]))
    )
    groups = {}
    for r in R:
      groups.setdefault(key(r), []).append(r)
    bad = sorted(
      ((pct(v, "success"), k, v) for k, v in groups.items() if pct(v, "success") < 90),
      key=lambda x: x[0],
    )
    print(f"  cells < 90 % success: {len(bad)} of {len(groups)}")
    for s, k, v in bad[:12]:
      print(
        f"    {s:3.0f}%  fall {pct(v, 'fall'):3.0f}%  "
        + (
          "h {:.2f} w {:.2f} off {:+.2f}".format(*k)
          if ph == "edge"
          else "h {:.2f} w {:.2f} vx {} ang {} run {}".format(*k)
        )
      )


conds = conditions()
rows = []
for fric in FRICS:
  if fric is not None:
    env.sim.model.geom_friction[:, wheel_gid, 0] = fric
  for b in range(0, len(conds), NENV):
    rows += [
      dict(r, fric=fric if fric is not None else "dr")
      for r in run_batch(conds[b : b + NENV], seed=b)
    ]
    print(
      f"  fric {fric}: batch {b // NENV + 1}/{math.ceil(len(conds) / NENV)} done",
      flush=True,
    )
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / f"{tag}.csv", "w", newline="") as fh:
  w = csv.DictWriter(fh, fieldnames=list(rows[0]))
  w.writeheader()
  w.writerows(rows)
if len(FRICS) == 1:
  summarize(rows)
else:
  print(
    f"\n### {ck.parent.name} @ {it}   by wheel friction (angles {ANG}), success/fall/stuck %"
  )
  phs = [ph for ph in ("bar", "edge", "up", "down") if ph in PHASES]
  print("  fric  " + "".join(f"{ph:>14s}" for ph in phs))
  for f in FRICS:
    print(
      f"  {f:4.2f}  "
      + "".join(
        f"{cell([r for r in rows if r['fric'] == f and r['phase'] == ph]):>14s}"
        for ph in phs
      )
    )
  for f in FRICS:
    print(f"\n======== wheel friction {f} ========")
    summarize([r for r in rows if r["fric"] == f])

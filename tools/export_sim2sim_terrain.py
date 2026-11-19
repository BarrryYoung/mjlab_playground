"""Export a task's training terrain to a unitree_mujoco scene for C++ sim2sim.

Builds the task's terrain generator exactly as in training (same sub-terrains,
proportions, rows, tile size, border), with a fixed seed, writes the terrain
(geoms + inline heightfields) into a scene that <include>s the robot xml, and
adds one keyframe per tile at the training spawn point (tile origin, base
0.45 m up, facing +x, default joint pos). In the simulate GUI: Simulation
panel -> Key -> Load key. Key 0 (also what reset goes back to) = the tile
picked by --home.

usage: python tools/export_sim2sim_terrain.py [--task T] [--seed S] [--home bar:0.40]
                                               [--robot go2w_std_m22.xml] [--out PATH]
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mjlab_playground  # noqa: F401  (registers the tasks)
import mujoco
import numpy as np
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator

SIM_DIR = Path.home() / "issac-sim-5/unitree_mujoco/unitree_robots/go2w"
# Per leg hip, thigh, calf, wheel (FL FR RL RR), the training default pose.
LEG_Q = "0 0.8 -1.5 0"
BASE_Z = 0.45

ap = argparse.ArgumentParser()
ap.add_argument(
  "--task", default="Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Unitree-Go2W"
)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--home", default="bar:0.40", help="<column>:<height>, tile for key 0")
ap.add_argument("--robot", default="go2w_std_m22.xml")
ap.add_argument("--out", default=str(SIM_DIR / "scene_climb_train_m22.xml"))
args = ap.parse_args()

cfg = load_env_cfg(args.task)
gen_cfg = replace(cfg.scene.terrain.terrain_generator, seed=args.seed, add_lights=False)
names = list(gen_cfg.sub_terrains)

# Record the sub-terrain and difficulty each tile was built with.
by_id = {id(sub): name for name, sub in gen_cfg.sub_terrains.items()}
tiles: dict[tuple[int, int], tuple[str, float]] = {}
_create = TerrainGenerator._create_terrain_geom


def _logged_create(self, spec, world_position, difficulty, cfg, sub_row, sub_col):
  tiles[sub_row, sub_col] = (by_id[id(cfg)], float(difficulty))
  return _create(self, spec, world_position, difficulty, cfg, sub_row, sub_col)


TerrainGenerator._create_terrain_geom = _logged_create

spec = mujoco.MjSpec()
gen = TerrainGenerator(gen_cfg)
gen.compile(spec)
origins = gen.terrain_origins  # (rows, cols, 3)
nrow, ncol = origins.shape[:2]


def obstacle_h(name: str, d: float) -> float | None:
  sub = gen_cfg.sub_terrains[name]
  if name == "bar":
    lo, hi = sub.height_range
    return lo + d * (hi - lo)
  if hasattr(sub, "step_height_range"):
    lo, hi = sub.step_height_range
    return lo + d * (hi - lo)
  return None


assert len(tiles) == nrow * ncol, (len(tiles), nrow, ncol)

keys = []
for col in range(ncol):
  for row in range(nrow):
    name, d = tiles[row, col]
    h = obstacle_h(name, d)
    label = f"{name}_r{row:02d}" + (f"_h{h:.2f}" if h is not None else f"_d{d:.2f}")
    o = origins[row, col]
    keys.append((label, name, h, o))

home_name, home_h = args.home.split(":")
cands = [k for k in keys if k[1] == home_name and k[2] is not None]
home = min(cands, key=lambda k: abs(k[2] - float(home_h)))
keys.remove(home)
keys.insert(0, home)

# Pull the terrain's assets and worldbody out of the generated MJCF. In-memory
# (buffer) textures cannot be written to XML: drop them, materials keep rgba.
for tex in list(spec.textures):
  spec.delete(tex)
for mat in spec.materials:
  mat.textures = [""] * len(mat.textures)
  mat.rgba = (0.55, 0.6, 0.55, 1.0)  # the heightfields' texture was their colour
root = ET.fromstring(spec.to_xml())
asset = root.find("asset")
world = root.find("worldbody")
# Shift the terrain so the key 0 tile sits at the world origin: the free camera
# looks there, and the GUI's Reset puts the robot at its xml pose (origin).
shift = -home[3] * np.array([1.0, 1.0, 0.0])
world.find("body[@name='terrain']").set("pos", " ".join(f"{v:g}" for v in shift))
keys = [(label, name, h, o + shift) for label, name, h, o in keys]

scene = ET.Element("mujoco", model="go2w climb training terrain")
ET.SubElement(scene, "include", file=args.robot)
# meansize: the viewer scales contact points / force arrows by it, and the
# metre-sized terrain boxes would otherwise blow it up (huge contact discs).
ET.SubElement(scene, "statistic", center="0 0 0.1", extent="0.8", meansize="0.05")
vis = ET.SubElement(scene, "visual")
ET.SubElement(
  vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0"
)
ET.SubElement(vis, "rgba", haze="0.15 0.25 0.35 1")
ET.SubElement(vis, "global", azimuth="-130", elevation="-20")
ET.SubElement(vis, "quality", shadowsize="4096")
a = ET.SubElement(scene, "asset")
ET.SubElement(
  a,
  "texture",
  type="skybox",
  builtin="gradient",
  rgb1="0.3 0.5 0.7",
  rgb2="0 0 0",
  width="512",
  height="3072",
)
if asset is not None:
  a.extend(list(asset))
wb = ET.SubElement(scene, "worldbody")
ET.SubElement(
  wb, "light", pos="0 0 30", dir="0 -0.3 -1", directional="true", castshadow="true"
)
wb.extend(list(world))
kf = ET.SubElement(scene, "keyframe")
for label, _, _, o in keys:
  q = f"{o[0]:.3f} {o[1]:.3f} {o[2] + BASE_Z:.3f} 1 0 0 0 " + " ".join([LEG_Q] * 4)
  ET.SubElement(kf, "key", name=label, qpos=q)

ET.indent(scene, space="  ")
lines = [
  f"  {i:3d}  {label}   x {o[0]:7.2f}  y {o[1]:7.2f}  z {o[2]:5.2f}"
  for i, (label, _, _, o) in enumerate(keys)
]
header = (
  f"<!-- Generated by mjlab_playground tools/export_sim2sim_terrain.py\n"
  f"     task {args.task}, seed {args.seed}: the training terrain, {nrow} rows (difficulty,\n"
  f"     +x) x {ncol} columns ({', '.join(names)}), tiles {gen_cfg.size[0]:g} m.\n"
  f"     Keys = training spawn points (Simulation panel -> Key -> Load key); key 0 is\n"
  f"     where reset goes. h = obstacle / step height (m), d = difficulty.\n"
  + "\n".join(lines)
  + "\n-->\n"
)
Path(args.out).write_text(header + ET.tostring(scene, encoding="unicode") + "\n")

# Check: it loads with the robot, and the keys fit.
m = mujoco.MjModel.from_xml_path(args.out)
dd = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, dd, 0)
mujoco.mj_forward(m, dd)
print(
  f"wrote {args.out}: {m.ngeom} geoms, {m.nhfield} hfields, {m.nkey} keys, nq {m.nq}"
)
print(f"key 0 = {keys[0][0]}")
for col in range(ncol):
  hs = [obstacle_h(*tiles[r, col]) for r in range(nrow)]
  rng = f"h {hs[0]:.2f}..{hs[-1]:.2f} m" if hs[0] is not None else "no height"
  print(f"  col {col} {names[col]:22s} {rng}")

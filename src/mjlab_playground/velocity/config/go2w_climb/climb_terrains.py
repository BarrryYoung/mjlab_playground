"""Extra climbing terrain for the Go2W Climb tasks: the long bar obstacle.

Target obstacle (11-10): a long bar 40 cm high and 40 cm wide (in the travel
direction). Getting over it without falling is the test; standing on it is not
required. The Go2W is longer than the bar is wide, so the front wheels are going
down the far side while the rear is still coming up the near side: up and down
in one motion, which the inverted-pyramid ledges (1 m treads) never ask for.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane


@dataclass(kw_only=True)
class BoxBarRingTerrainCfg(SubTerrainCfg):
  """Flat floor at z = 0 with a square ring wall around the spawn point.

  The wall's inner face is ``ring_distance`` from the tile centre, it is
  ``bar_width`` thick and ``h`` high, and each side is 2 * (ring_distance +
  bar_width) long, so whichever way the robot leaves the centre it meets a long
  bar square on. Past the bar the floor is flat again; the stock distance
  curriculum (promote beyond half a tile, 4 m) therefore needs the bar crossed.

  Height: ``heights`` (one per row, exact; for benchmarks) if given, else
  linear in difficulty over ``height_range``. Width: uniform in
  ``bar_width_range`` per tile.
  """

  height_range: tuple[float, float] = (0.05, 0.50)
  heights: tuple[float, ...] | None = None
  bar_width_range: tuple[float, float] = (0.3, 0.5)
  ring_distance: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    if self.heights:
      h = self.heights[min(int(difficulty * len(self.heights)), len(self.heights) - 1)]
    else:
      lo, hi = self.height_range
      h = lo + difficulty * (hi - lo)
    w = float(rng.uniform(*self.bar_width_range))
    a = self.ring_distance
    cx, cy = 0.5 * self.size[0], 0.5 * self.size[1]

    geometries = [
      TerrainGeometry(geom=g, color=(0.5, 0.5, 0.5, 1.0))
      for g in make_plane(body, self.size, 0.0, center_zero=False)
    ]
    rgba = (0.85, 0.55, 0.20, 1.0)
    for s in (-1.0, 1.0):
      # Bars across x (the robot meets them driving along +-x), full length.
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(w / 2, a + w, h / 2),
        pos=(cx + s * (a + w / 2), cy, h / 2),
      )
      geometries.append(TerrainGeometry(geom=g, color=rgba))
      # Bars across y, between the x bars.
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(a, w / 2, h / 2),
        pos=(cx, cy + s * (a + w / 2), h / 2),
      )
      geometries.append(TerrainGeometry(geom=g, color=rgba))
    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geometries)

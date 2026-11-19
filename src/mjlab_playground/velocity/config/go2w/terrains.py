"""Go2W high-platform terrain: climb from 5 cm up to 50 cm.

Mixed terrain after the M20 high-platform recipe (Dreamwaq ``highplatform``
branch), with the climbing tiles as one column among ordinary rough terrain.

In mjlab curriculum mode every sub-terrain gets exactly one column and
``proportion`` only sets how many envs spawn on it. Difficulty goes along the
rows: row ``r`` draws its difficulty uniformly from ``[r, r+1) / num_rows``.

Run this file to build the grid, print the per-row heights and open a viewer.
"""

import mjlab.terrains as terrain_gen
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)

HIGH_PLATFORM_NAME = "high_platform"

# 10 rows over 0.05-0.50 m: 4.5 cm per level.
HIGH_PLATFORM_STEP_RANGE = (0.05, 0.50)

HIGHPLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=7,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.10),
    "random_rough": random_rough(proportion=0.10, noise_range=(0.02, 0.06)),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.05, slope_range=(0.0, 0.4)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.05, slope_range=(0.0, 0.4)
    ),
    "stairs_up": pyramid_stairs(proportion=0.20, step_height_range=(0.02, 0.12)),
    "stairs_down": pyramid_stairs_inv(proportion=0.20, step_height_range=(0.02, 0.12)),
    # The climb. Inverted, so the robot spawns on the low centre floor with
    # each rise followed by a flat tread to land and recover on.
    #
    # ``platform_width`` does not set the centre floor size, it only feeds
    #     num_steps   = int((size - 2*border - platform_width) / (2*step_width))
    # while the floor the robot spawns on is
    #     centre side = size - 2*border - 2*num_steps*step_width
    # Here: num_steps = int((8 - 2 - 3) / 2) = 1, centre = 8 - 2 - 2 = 4 m.
    #
    # Layout from the centre out: 2 m of floor, rise, 1 m tread, rise, 1 m
    # border at tile height 0. Two rises of ``step_height`` each. The stock
    # distance curriculum promotes at > 4 m, i.e. only after both rises.
    HIGH_PLATFORM_NAME: terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.30,
      step_height_range=HIGH_PLATFORM_STEP_RANGE,
      step_width=1.0,
      platform_width=3.0,
      border_width=1.0,
    ),
  },
  add_lights=True,
)


if __name__ == "__main__":
  import mujoco.viewer
  import torch
  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

  cfg = HIGHPLATFORM_TERRAINS_CFG
  lo, hi = HIGH_PLATFORM_STEP_RANGE
  print(f"{HIGH_PLATFORM_NAME} step height per row (two rises each):")
  for r in range(cfg.num_rows):
    a = lo + (hi - lo) * r / cfg.num_rows
    b = lo + (hi - lo) * (r + 1) / cfg.num_rows
    print(f"  row {r}: {a:.3f} - {b:.3f} m")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  terrain = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=cfg), device=device
  )
  print("terrain_origins (rows x cols x 3), z:")
  print(terrain.terrain_origins[..., 2].cpu().numpy().round(3))
  mujoco.viewer.launch(terrain.spec.compile())

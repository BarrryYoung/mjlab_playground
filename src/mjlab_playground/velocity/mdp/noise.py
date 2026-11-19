from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.noise import UniformNoiseCfg


@dataclass
class EnvSplitUniformNoiseCfg(UniformNoiseCfg):
  """UniformNoiseCfg that leaves the first ``clean_frac`` of the envs noiseless.

  The split is by env index, so each env keeps its noisy / clean role for the
  whole run. Clean envs see the observations the deployed policy sees; with
  the default noise the stepping loop that keeps the robot trotting after a
  zero command is broken by the noise and so never shows up in training.
  """

  clean_frac: float = 0.0

  def apply(self, data: torch.Tensor) -> torch.Tensor:
    noisy = super().apply(data)
    k = int(round(data.shape[0] * self.clean_frac))
    if k > 0:
      noisy[:k] = data[:k]
    return noisy

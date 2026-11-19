"""Recurrent actor that warm-starts from an MLP actor.

rsl_rl's RNNModel puts the GRU in front of the MLP (obs -> GRU -> MLP), so an
MLP checkpoint cannot be loaded into it: the MLP's first layer would see the GRU
state instead of the observation. Here the GRU runs alongside and its state is
appended to the observation (obs -> GRU -> h; MLP([obs, h])), the same layout
as appending terms to the input. The MLP's first layer gets zero columns for h,
so a warm-started model computes exactly what the MLP did, whatever the GRU
weights are; the zero columns get gradient (dL/dW = delta * h) and grow.

Everything else (training, hidden-state bookkeeping, ONNX with h_in / h_out) is
rsl_rl's RNNModel.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class SkipRNNModel(RNNModel):
  """obs -> GRU / LSTM -> h; MLP head on [obs, h]."""

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    x = MLPModel.get_latent(self, obs)
    h = self.rnn(x, masks, hidden_state).squeeze(0)
    if masks is not None:
      # Batch mode: the RNN returns unpadded [time, batch, H] for the valid
      # steps; the observations come padded, so unpad them the same way.
      from rsl_rl.utils import unpad_trajectories

      x = unpad_trajectories(x, masks)
    return torch.cat([x, h], dim=-1)

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.latent_dim

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("SkipRNNModel: use ONNX export")

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxSkipRNNModel(self, verbose)


class _OnnxSkipRNNModel(nn.Module):
  """ONNX: (obs [1, D], h_in [L, 1, H]) -> (actions, h_out), as rsl_rl's GRU export."""

  is_recurrent: bool = True

  def __init__(self, model: SkipRNNModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    if not isinstance(self.rnn, nn.GRU):
      raise NotImplementedError("SkipRNNModel ONNX export: GRU only")
    self.input_size = model.obs_dim
    self.hidden_size = self.rnn.hidden_size
    self.num_layers = self.rnn.num_layers

  def forward(self, obs: torch.Tensor, h_in: torch.Tensor):
    x = self.obs_normalizer(obs)
    h, h_out = self.rnn(x.unsqueeze(0), h_in)
    out = self.mlp(torch.cat([x, h.squeeze(0)], dim=-1))
    return self.deterministic_output(out), h_out

  def get_dummy_inputs(self):
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(self.num_layers, 1, self.hidden_size),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]

"""Stand displacement (11-24 晚): 0 指令期间相对“开始站立那一刻”的位置的水平位移。

- 锚点：指令从非 0 变成 0（或 episode 开始时指令就是 0）后先“等停”，机身水平速度 < v_stop
  （5 cm/s）或等满 timeout_s（1.5 s）时才记锚点。刹车距离不计入（交给速度跟踪 / stand_still），
  只罚停下来之后的溜车；超时防止靠一直慢慢挪来逃避。指令保持 0 时重采样到 0 不换锚点。
  等停期间惩罚和观测都是 0。
- 奖励 stand_displacement_penalty：0 指令时 min(d, max_dist) / scale，非 0 指令时 0，配负权重。
  线性、不饱和到 max_dist，接近 0 时梯度不消失（exp 核在小误差处是平的）。
- 观测 stand_displacement_b：机体系下的 (dx, dy)，非 0 指令时为 0；只给 critic（同 heading_err），
  让价值函数能预测这项惩罚。actor 不需要看到它：0 指令时把轮子输出稳在 0 就不会溜。

状态存在 env 上，奖励和观测都调用 _update，函数幂等：同一步调用多次结果一样；
reset 后 episode_length_buf == 0 的环境在观测计算时重设锚点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ATTR = "_stand_disp_state"


V_STOP = 0.05  # m/s
TIMEOUT_S = 1.5


def _update(env: ManagerBasedRlEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
  """returns (active [N] bool: 0 指令且已记锚点, displacement_w [N, 2])。

  每步会被奖励（reset 前）和观测（reset 后）各调用一次：计时器按 common_step_counter 每步只走一次，
  其余逻辑重复调用结果不变。"""
  robot = env.scene["robot"]
  pos = robot.data.root_link_pos_w[:, :2]
  speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=1)
  cmd = env.command_manager.get_command(command_name)
  zero = cmd[:, :3].abs().sum(dim=1) == 0.0
  st = getattr(env, _ATTR, None)
  if st is None:
    f = torch.zeros_like(zero)
    st = {"anchor": pos.clone(), "prev": f.clone(), "waiting": f.clone(), "active": f.clone(),
          "wait": torch.zeros(zero.shape[0], dtype=torch.long, device=zero.device), "step": -1}
    setattr(env, _ATTR, st)
  start = zero & (~st["prev"] | (env.episode_length_buf == 0))
  st["waiting"] |= start
  st["active"] &= ~start
  st["wait"][start] = 0
  if st["step"] != env.common_step_counter:
    st["step"] = env.common_step_counter
    st["wait"] += st["waiting"].long()
  timeout = int(round(TIMEOUT_S / env.step_dt))
  fire = st["waiting"] & ((speed < V_STOP) | (st["wait"] >= timeout))
  st["anchor"][fire] = pos[fire]
  st["waiting"] &= ~fire & zero
  st["active"] = (st["active"] | fire) & zero
  st["prev"] = zero.clone()
  disp = (pos - st["anchor"]) * st["active"].unsqueeze(-1).float()
  return st["active"], disp


def stand_displacement_penalty(
  env: ManagerBasedRlEnv, command_name: str, scale: float = 0.1, max_dist: float = 0.5
) -> torch.Tensor:
  zero, disp = _update(env, command_name)
  d = disp.norm(dim=1).clamp(max=max_dist)
  return d / scale * zero.float()


def stand_displacement_b(
  env: ManagerBasedRlEnv, command_name: str, max_dist: float = 0.5
) -> torch.Tensor:
  _, disp = _update(env, command_name)
  yaw = env.scene["robot"].data.heading_w
  c, s = torch.cos(yaw), torch.sin(yaw)
  dx = c * disp[:, 0] + s * disp[:, 1]
  dy = -s * disp[:, 0] + c * disp[:, 1]
  return torch.stack([dx, dy], dim=-1).clamp(-max_dist, max_dist)

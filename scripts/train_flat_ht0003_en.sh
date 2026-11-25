#!/usr/bin/env bash
# Flat Dgv2-M22-LowNoise，lownoise_ht0003 配方 + leg_energy。
# 覆盖参数与 11-20 的 ht0003push10-m22-lownoise-en*-n2048 完全一致（逐项 diff 过 env.yaml）。
# task 默认值和这套配方差很多（dgv2=0、髋惩罚=0 等），不能只传 leg_energy。
# 用法: bash scripts/train_flat_ht0003_en.sh <leg_energy权重> <run_name> [num_envs]
set -euo pipefail
cd "$(dirname "$0")/.."

W="${1:?leg_energy weight, e.g. -5e-4}"
RUN="${2:?run name}"
N="${3:-4096}"

ARGS=(
  --env.scene.num-envs "$N"
  --agent.max-iterations 8000
  --env.rewards.track-linear-velocity.weight 1.0
  --env.rewards.track-linear-velocity.params.cmd-scaled True
  --env.rewards.track-linear-velocity.params.std-min 0.1
  --env.rewards.track-angular-velocity.weight 1.0
  --env.rewards.track-angular-velocity.params.cmd-scaled True
  --env.rewards.track-angular-velocity.params.std-min 0.1
  --env.rewards.upright.weight 0.5
  --env.rewards.pose.weight 0.5
  --env.rewards.dof-pos-limits.weight 0
  --env.rewards.action-rate-l2.weight -0.01
  --env.rewards.foot-slip.weight 0
  --env.rewards.soft-landing.weight 0
  --env.rewards.foot-clearance.weight 0
  --env.rewards.stand-still-wheel-avg.weight -0.03
  --env.rewards.foot-swing-clearance.weight -0.35
  --env.rewards.dgv2.weight 0.5
  --env.rewards.dgv2.params.balance-power 2.0
  --env.rewards.dgv2.params.require-all-down True
  --env.rewards.hip-torques-l2.weight -0.003
  --env.events.push-robot.interval-range-s "(9.0, 11.0)"
  "--env.rewards.leg-energy.weight=$W"
  --agent.run-name "$RUN"
)

exec .venv/bin/train "${FLAT_TASK:-Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise}" "${ARGS[@]}"

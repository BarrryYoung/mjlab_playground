#!/usr/bin/env bash
# Flat Dgv2-M22-LowNoise + 凸包轮子，lownoise_ht0003 配方（髋惩罚 -0.003，不带能量项）。
# 用法: bash scripts/train_hullwheel.sh [额外 tyro 参数...]
# 一行一个参数，避免从终端复制长行时被折断。
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=(
  --env.scene.num-envs 2048
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
  --agent.run-name tl1ta1p05a001fsc035-004u05sa003dgv2w05bp2alldownht0003-m22-av005jv05-n2048-hullwheel-scratch
)

exec uv run train Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise-HullWheel \
  "${ARGS[@]}" "$@"

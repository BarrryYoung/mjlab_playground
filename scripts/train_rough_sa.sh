#!/usr/bin/env bash
# Rough SA：只把 stand_still_wheel_avg 权重放大（x5=-0.15 / x10=-0.3），针对 0 指令溜车。
# 从 11-19 ht0003(2048 env) model_13998 续训 1000 iter，2048 env，课程列从最高排开始，
# 与 ZeroStd/WheelPos 续训实验条件一致。热启动 = 原 ckpt 拷贝（actor 342 不变）。
# 用法: bash scripts/train_rough_sa.sh 5|10
set -euo pipefail
cd "$(dirname "$0")/.."
M="${1:?5|10}"
exec .venv/bin/train "Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SA${M}-Unitree-Go2W" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.resume True \
  --agent.load-run warmstart_ht0003_13998 \
  --agent.load-checkpoint model_13998.pt \
  --agent.run-name "straight-h6-lownoise-ht0003-sa${M}x-n2048-ws-ht0003-13998"

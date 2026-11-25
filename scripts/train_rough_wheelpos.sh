#!/usr/bin/env bash
# Rough Straight-H6-LowNoise-Ht0003 + wheel_vel_pos（轮子转角差 5 步 / 0.1 s，actor 342 -> 346）。
# 从 11-19 的 ht0003 run(2048 env) model_13998 热启动，续训 1000 iter，2048 env 与原 run 一致。
# 热启动权重: tools/warmstart_ckpt.py 生成在 warmstart_ht0003_13998/（旧 342 列原样，新 4 列为 0）。
# 用法: bash scripts/train_rough_wheelpos.sh [run_name] [num_envs] [max_iterations]
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-straight-h6-lownoise-ht0003-wheelpos5-n2048-ws-ht0003-13998}"
N="${2:-2048}"
ITERS="${3:-1000}"

exec uv run train Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-WheelPos-Unitree-Go2W \
  --env.scene.num-envs "$N" \
  --agent.max-iterations "$ITERS" \
  --agent.resume True \
  --agent.load-run warmstart_ht0003_13998 \
  --agent.load-checkpoint model_13998.pt \
  --agent.run-name "$RUN"

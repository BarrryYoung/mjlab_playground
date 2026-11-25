#!/usr/bin/env bash
# Rough ZeroStd：0 指令时速度跟踪 std 也用 0.1（原来回到 0.5/0.71）。
# 从 11-19 的 ht0003 run(2048 env) model_13998 续训 1000 iter，2048 env 与原 run 一致，
# 课程列从最高排开始。
#   plain    : 只改奖励（actor 342），热启动 = 原 ckpt 拷贝（warmstart_ht0003_13998/）
#   wheelpos : 改奖励 + wheel_vel_pos（actor 346），热启动 = warmstart_ckpt.py 扩列后，
#              把 joint_vel 最新帧 4 个轮子列权重对半分给新 4 列（warmstart_ht0003_13998_split/）
# 用法: bash scripts/train_rough_zerostd.sh plain|wheelpos [num_envs] [max_iterations]
set -euo pipefail
cd "$(dirname "$0")/.."

V="${1:?plain|wheelpos}"
N="${2:-2048}"
ITERS="${3:-1000}"
case "$V" in
  plain)
    TASK=Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-ZeroStd-Unitree-Go2W
    LOAD=warmstart_ht0003_13998
    RUN=straight-h6-lownoise-ht0003-zerostd01-n2048-ws-ht0003-13998 ;;
  wheelpos)
    TASK=Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-ZeroStd-WheelPos-Unitree-Go2W
    LOAD=warmstart_ht0003_13998_split
    RUN=straight-h6-lownoise-ht0003-zerostd01-wheelpos5split-n2048-ws-ht0003-13998 ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac

exec uv run train "$TASK" \
  --env.scene.num-envs "$N" \
  --agent.max-iterations "$ITERS" \
  --agent.resume True \
  --agent.load-run "$LOAD" \
  --agent.load-checkpoint model_13998.pt \
  --agent.run-name "$RUN"

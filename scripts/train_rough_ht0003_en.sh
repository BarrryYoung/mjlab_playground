#!/usr/bin/env bash
# rough straight-h6 lownoise_ht0003 + flat 那套能量惩罚(leg_energy)。
# 用法: bash scripts/train_rough_ht0003_en.sh <leg_energy权重> <run_name> [num_envs] [load_run] [load_ckpt]
#   权重 0 = 加这一项但权重为 0（基线，实际等于原配置）
#   load_run 默认 warmstart_ht0003；续跑时传自己那个 run 目录名，load_ckpt 传最新的 model_XXXX.pt
set -euo pipefail
cd "$(dirname "$0")/.."

W="${1:?leg_energy weight, e.g. -5e-4}"
RUN="${2:?run name}"
N="${3:-4096}"
LOAD_RUN="${4:-warmstart_ht0003}"
LOAD_CKPT="${5:-model_7999.pt}"

# 四组统一：6000 iter、4096 env、默认从 warmstart_ht0003/model_7999.pt 热启动。
ARGS=(
  --env.scene.num-envs "$N"
  --agent.max-iterations 6000
  --agent.resume True
  --agent.load-run "$LOAD_RUN"
  --agent.load-checkpoint "$LOAD_CKPT"
  --agent.run-name "$RUN"
)
# 用 = 形式，避免 tyro 把负号当成选项。
if [ "$W" != "0" ]; then
  ARGS+=("--env.rewards.leg-energy.weight=$W")
fi

exec .venv/bin/train "${ROUGH_TASK:-Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Unitree-Go2W}" "${ARGS[@]}"

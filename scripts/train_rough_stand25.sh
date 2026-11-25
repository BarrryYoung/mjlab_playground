#!/usr/bin/env bash
# Rough Stand25：平地类地形 standing 桶 10% -> 25%（slow、combo 各 30% -> 22.5%），其他同 ht0003。
# 对照 11-19 ht0003：从 flat warmstart_ht0003/model_7999.pt 完整训练 6000 iter，2048 env，课程从低排开始。
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/train Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Stand25-Unitree-Go2W \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run warmstart_ht0003 \
  --agent.load-checkpoint model_7999.pt \
  --agent.run-name straight-h6-lownoise-ht0003-stand25-n2048-ws-ht0003

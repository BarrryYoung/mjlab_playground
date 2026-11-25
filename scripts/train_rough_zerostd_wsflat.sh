#!/usr/bin/env bash
# Rough ZeroStd+WheelPos，从 flat 热启动完整训练 6000 iter（7999 -> 13998），不从已溜车的 rough ckpt 续训。
# 课程从低排开始，与原 rough run 一致；热启动 = warmstart_ckpt.py 342->346 + split_wheelpos_init.py 对半分。
#   ht0003 : 对照 11-19 ht0003（leg_energy 0，2048 env，warmstart_ht0003）
#   en5e4  : 对照 11-24 13-24-29 en5e4（leg_energy -5e-4，4096 env，warmstart_flat_en5e4_n4096）
# 用法: bash scripts/train_rough_zerostd_wsflat.sh ht0003|en5e4
set -euo pipefail
cd "$(dirname "$0")/.."
P=Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003
case "${1:?ht0003|en5e4}" in
  ht0003) TASK=$P-ZeroStd-WheelPos-WsFlat-Unitree-Go2W; N=2048; LOAD=warmstart_ht0003_split
          RUN=straight-h6-lownoise-ht0003-zerostd01-wheelpos5split-n2048-ws-ht0003 ;;
  en5e4)  TASK=$P-En5e4-ZeroStd-WheelPos-WsFlat-Unitree-Go2W; N=4096; LOAD=warmstart_flat_en5e4_n4096_split
          RUN=straight-h6-lownoise-ht0003-en5e4-zerostd01-wheelpos5split-n4096-ws-flaten5e4n4096 ;;
  *) echo "unknown $1"; exit 1 ;;
esac
exec .venv/bin/train "$TASK" \
  --env.scene.num-envs "$N" \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run "$LOAD" \
  --agent.load-checkpoint model_7999.pt \
  --agent.run-name "$RUN"

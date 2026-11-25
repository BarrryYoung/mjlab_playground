#!/usr/bin/env bash
# SaDisp 完整重训：flat 从零（11-17 flat ht0003 配方，2048 env，8000 iter）-> rough（11-19 ht0003 配方，
# 2048 env，6000 iter）。两段都保留 stand_still_wheel_avg -0.03，加 0 指令位移惩罚 + critic 位移观测。
# 对照：flat 11-17_11-29-26 @7999 / rough 11-19_11-59-27 @13998。同一脚本顺序执行。
set -euo pipefail
cd "$(dirname "$0")/.."
FLAT_TASK=Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise-SaDisp
ROUGH_TASK=Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SaDisp-Unitree-Go2W
FLAT_RUN=ht0003push10-m22-lownoise-sadisp-n2048-scratch
FLAT_DIR=logs/rsl_rl/playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_sadisp_velocity
ROUGH_DIR=logs/rsl_rl/playground_go2w_rough_straight_h6_lownoise_ht0003_sadisp_velocity
WS=warmstart_flat_sadisp_n2048
ROUGH_RUN=straight-h6-lownoise-ht0003-sadisp-n2048-ws-flatsadisp

echo "[chain] $(date) flat start"
FLAT_TASK=$FLAT_TASK bash scripts/train_flat_ht0003_en.sh 0 "$FLAT_RUN" 2048
CK=$(ls -td "$FLAT_DIR"/*_"$FLAT_RUN" | head -1)/model_7999.pt
[ -f "$CK" ] || { echo "[chain] missing $CK"; exit 1; }
mkdir -p "$ROUGH_DIR/$WS"
.venv/bin/python tools/warmstart_ckpt.py "$CK" --task "$ROUGH_TASK" --out "$ROUGH_DIR/$WS/model_7999.pt"
echo "[chain] $(date) warmstart <- $CK"
ROUGH_TASK=$ROUGH_TASK exec bash scripts/train_rough_ht0003_en.sh 0 "$ROUGH_RUN" 2048 "$WS" model_7999.pt

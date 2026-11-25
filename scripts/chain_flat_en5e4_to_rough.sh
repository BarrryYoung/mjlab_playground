#!/usr/bin/env bash
# 1) flat lownoise ht0003 + leg_energy -5e-4，4096 env，从零 8000 iter
# 2) 跑完后把它的 model_7999.pt 拷到 rough 的 warmstart_flat_en5e4_n4096/
# 3) rough straight-h6 lownoise ht0003 从这个 warmstart 热启动，leg_energy 同为 -5e-4
# 同一脚本里顺序执行，不靠 pgrep 等待。
set -euo pipefail
cd "$(dirname "$0")/.."

FLAT_TASK=Playground-Velocity-Flat-Unitree-Go2W-IsaacObs-Cmd-Dgv2-M22-LowNoise
FLAT_RUN=ht0003push10-m22-lownoise-en5e4-n4096
FLAT_DIR=logs/rsl_rl/playground_go2w_flat_isaacobs_cmd_dgv2_m22_lownoise_velocity
ROUGH_DIR=logs/rsl_rl/playground_go2w_rough_straight_h6_lownoise_ht0003_velocity
WS=warmstart_flat_en5e4_n4096
ROUGH_RUN=straight-h6-lownoise-ht0003-en5e4-n4096-ws-flaten5e4n4096

echo "[chain] $(date) flat start"
bash scripts/train_flat_ht0003_en.sh -5e-4 "$FLAT_RUN" 4096

CK=$(ls -td "$FLAT_DIR"/*_"$FLAT_RUN" | head -1)/model_7999.pt
[ -f "$CK" ] || { echo "[chain] missing $CK"; exit 1; }
mkdir -p "$ROUGH_DIR/$WS"
# flat actor 57 维 -> rough 342 维，必须用 warmstart_ckpt.py 转，不能直接 cp
.venv/bin/python tools/warmstart_ckpt.py "$CK" \
  --task Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-Unitree-Go2W \
  --out "$ROUGH_DIR/$WS/model_7999.pt"
echo "[chain] $(date) warmstart <- $CK"

exec bash scripts/train_rough_ht0003_en.sh -5e-4 "$ROUGH_RUN" 4096 "$WS" model_7999.pt

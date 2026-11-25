#!/usr/bin/env bash
# SaDisp 续训：11-19 ht0003 @13998 续训 1000 iter，2048 env，课程列从最高排开始（同 SA/ZeroStd 续训）。
# critic 追加 stand_disp(2)，热启动 = warmstart_ckpt.py 273->275（actor 不变）。
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/train Playground-Velocity-Rough-Straight-H6-LowNoise-Ht0003-SaDisp-Resume-Unitree-Go2W \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.resume True \
  --agent.load-run warmstart_ht0003_13998 \
  --agent.load-checkpoint model_13998.pt \
  --agent.run-name straight-h6-lownoise-ht0003-sadisp-n2048-ws-ht0003-13998

#!/usr/bin/env bash
# 爬台进入角随机化：v6b (Bar2)@8500 续训 2000 iter，爬台列朝向噪声 ±10° -> ±30°，其余同 v6b（4096 env，课程起点 25 级）。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
exec .venv/bin/train Playground-Velocity-Climb-Blind-DC-H65-FlatPose-Bar2-Angle30-Unitree-Go2W \
  --env.scene.num-envs 4096 \
  --env.scene.terrain.max-init-terrain-level 25 \
  --agent.max-iterations 2000 \
  --agent.resume True \
  --agent.load-run 2026-11-11_00-45-08_climb-v6b-flatpose-bar2 \
  --agent.load-checkpoint model_8500.pt \
  --agent.run-name climb-v6b-angle30-ws8500

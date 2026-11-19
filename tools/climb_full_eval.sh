#!/usr/bin/env bash
# Usage: tools/climb_full_eval.sh <task> <run-glob@iter> <logdir-name> <tag>
# Runs climb success (full range + 0.40-0.65 m), flat tracking/stand drift, CPU sim2sim; writes /tmp/climb_evals/<tag>.txt
set -u
cd ~/debug/my_mjlab_playground/mjlab_playground && source .venv/bin/activate
T_=$1; S=$2; L=$3; O=/tmp/climb_evals/$4.txt
F='warn|term_cfg'
{
echo "### $T_ $S  $(date +%H:%M)"
echo "-- climb full range"; E=16 T=14 python tools/climb_eval.py "$T_" "$S" 2>&1 | grep -viE "$F" | grep -E "overall"
echo "-- climb 0.40-0.65"; HRANGE=0.40,0.65 E=16 T=14 python tools/climb_eval.py "$T_" "$S" 2>&1 | grep -viE "$F" | tail -28
echo "-- flat"; python tools/climb_flat.py "$T_" "$S" "$L" 2>&1 | grep -viE "$F" | tail -6
echo "-- sim2sim cpu"; for y in 0 6; do YAW_DEG=$y H=0.4,0.5,0.55,0.6 T=12 timeout 1200 python tools/climb_sim2sim_cpu.py "$T_" "$S" 2>&1 | grep -viE "$F" | grep -E "^  h|Error" | sed "s/^/yaw $y: /"; done
} > "$O" 2>&1
echo done "$O"

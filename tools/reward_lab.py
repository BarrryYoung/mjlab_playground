"""Kept so old `uv run python tools/reward_lab.py ...` commands still work.

The tool now lives in mjlab_playground.reward_lab; prefer `uv run rewardlab`.
"""

from mjlab_playground.reward_lab import main

if __name__ == "__main__":
  main()

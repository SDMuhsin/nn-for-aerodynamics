#!/usr/bin/env bash
# Print the list of CUDA device identifiers that scripts/sweep.py should use.
#
# - If the host has MIG instances, print one MIG UUID per line.
# - Else, print one integer per physical GPU index.
#
# Used by scripts/run_full_sweep.sh; also handy to double-check the layout
# before launching a long sweep:
#
#   bash scripts/detect_gpus.sh
#
# Honors GEOMPNN_GPUS (whitespace-separated identifiers) as an explicit
# override, e.g.
#
#   GEOMPNN_GPUS="MIG-aaa MIG-bbb 0" bash scripts/run_full_sweep.sh

set -eo pipefail

if [[ -n "${GEOMPNN_GPUS:-}" ]]; then
  for g in $GEOMPNN_GPUS; do echo "$g"; done
  exit 0
fi

# nvidia-smi -L produces one line per device. MIG layout looks like:
#   GPU 0: NVIDIA RTX PRO 6000 Black (UUID: GPU-xxx)
#     MIG 1g.24gb     Device  0: (UUID: MIG-yyy)
# Plain GPU layout looks like:
#   GPU 0: NVIDIA A40 (UUID: GPU-aaa)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "0"  # CPU-only fallback (won't actually train, but lets sweep loop)
  exit 0
fi

mapfile -t MIG_UUIDS < <(nvidia-smi -L 2>/dev/null \
  | grep -oE 'MIG-[0-9a-fA-F-]+' \
  | sort -u)

if [[ ${#MIG_UUIDS[@]} -gt 0 ]]; then
  for u in "${MIG_UUIDS[@]}"; do echo "$u"; done
  exit 0
fi

# No MIG: list integer GPU indices
nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null

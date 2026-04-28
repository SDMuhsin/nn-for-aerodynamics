#!/usr/bin/env bash
# Turn-key driver for the full GeoMPNN reproduction sweep.
#
#   bash scripts/run_full_sweep.sh                 # default: all 10 ablations × 8 seeds
#   bash scripts/run_full_sweep.sh --ablations s2v geompnn   # subset
#   bash scripts/run_full_sweep.sh --seeds 0 1 2   # subset
#   GEOMPNN_GPUS="MIG-uuid1 MIG-uuid2" bash scripts/run_full_sweep.sh   # pin slots
#
# What it does
# ------------
# 1. Activates ./env (created by scripts/setup.sh)
# 2. Auto-detects GPU/MIG slots via scripts/detect_gpus.sh
# 3. Tee's all stdout/stderr to logs/run_full_sweep_<timestamp>.log
# 4. Launches scripts/sweep.py with one parallel slot per detected device
# 5. Aggregates final results to results/summary.md when the sweep finishes
#
# ETA
# ---
# scripts/sweep.py prints a "FINAL ETA" line at startup AND on every per-run
# completion AND every 5 minutes. Before any run completes it uses a 9 h/run
# placeholder (our measured A40 baseline). Once runs finish it switches to the
# observed mean wall-clock per run.

# NOTE: NO `set -u`. CC's Lmod `module` bash function references unset vars
# internally; under nounset it silently aborts. See scripts/setup_cc.sh header
# and the round-2/round-3 install diagnostics.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Defaults — currently scoped to ONLY the paper's headline SOTA model (geompnn,
# target Global Score ~53–55, must beat Casenave 47.04). The other 9 ablations
# below form the full reproduction sweep; uncomment when ready to run them.
ABLATIONS=(geompnn)
# ABLATIONS=(mlp gnn s2v s2v_gnn trail polar sine sph inlet geompnn)
SEEDS=(0 1 2 3 4 5 6 7)
ETA_INTERVAL_SEC=300
ETA_SEED_HOURS=9.0
EXTRA_SWEEP_ARGS=()

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

# Argument parsing — collect optional --ablations / --seeds and pass everything
# else to scripts/sweep.py verbatim (so e.g. --skip_eval still works).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ablations)
      shift
      ABLATIONS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do ABLATIONS+=("$1"); shift; done ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --eta_interval)
      ETA_INTERVAL_SEC="$2"; shift 2 ;;
    --eta_seed_hours)
      ETA_SEED_HOURS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *)
      EXTRA_SWEEP_ARGS+=("$1"); shift ;;
  esac
done

# Activate venv. On CC, also load the modules that provide vtk's Python
# bindings (we can't pip-install vtk for cp311; see setup_cc.sh header).
if [[ ! -d env ]]; then
  echo "[run] no ./env found. Run: bash scripts/setup.sh"
  exit 2
fi
if [[ -d /cvmfs/soft.computecanada.ca ]]; then
  module load StdEnv/2023 gcc/12.3 python/3.11 vtk/9.3.0
fi
source env/bin/activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="$ROOT/data/hf"
export TORCH_HOME="$ROOT/data/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Detect GPU slots
mapfile -t GPUS < <(bash "$SCRIPT_DIR/detect_gpus.sh")
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[run] no GPUs detected (and GEOMPNN_GPUS not set). Aborting."
  exit 3
fi
echo "[run] $(date '+%Y-%m-%d %H:%M:%S')  detected ${#GPUS[@]} slot(s):"
for g in "${GPUS[@]}"; do echo "      $g"; done

# Sanity: AirfRANS data must be present
if [[ ! -f "$ROOT/data/airfrans/Dataset/manifest.json" ]]; then
  echo "[run] AirfRANS dataset missing at data/airfrans/Dataset/."
  echo "[run] Run: bash scripts/setup.sh   (or pass --no-data and download manually)"
  exit 4
fi

# Logging
mkdir -p logs
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="$ROOT/logs/run_full_sweep_${TS}.log"
echo "[run] full log: $LOG"

# Initial estimate banner — gives the user a "FINAL ETA" line BEFORE the first
# Python import even runs, in case they're checking the log immediately.
TOTAL_RUNS=$(( ${#ABLATIONS[@]} * ${#SEEDS[@]} ))
SLOTS=${#GPUS[@]}
INIT_CYCLES=$(( (TOTAL_RUNS + SLOTS - 1) / SLOTS ))
# Date arithmetic via python (portable across distros; venv is already sourced).
read -r INIT_ETA_HUMAN INIT_ETA_CLOCK < <(python - <<PY
import datetime as dt
total_h = $INIT_CYCLES * $ETA_SEED_HOURS
d, rem_h = divmod(total_h, 24)
human = f"{int(d)}d {rem_h:.1f}h" if d else f"{total_h:.1f}h"
clock = (dt.datetime.now() + dt.timedelta(hours=total_h)).strftime("%Y-%m-%d %H:%M:%S")
print(human, clock)
PY
)
{
  echo "[run] ablations: ${ABLATIONS[*]}"
  echo "[run] seeds:     ${SEEDS[*]}"
  echo "[run] runs:      $TOTAL_RUNS  slots: $SLOTS"
  echo "[run] FINAL ETA (initial estimate at ${ETA_SEED_HOURS}h/run): ${INIT_ETA_HUMAN}  (~${INIT_ETA_CLOCK})"
  echo "[run] sweep.py will refine this every ${ETA_INTERVAL_SEC}s and on every completion."
} | tee -a "$LOG"

# Launch the sweep.
# `unbuffer` would be nicer but we don't want to require expect; use python -u
# and stdbuf so prints land in the log promptly.
set +e
stdbuf -oL -eL python -u scripts/sweep.py \
    --ablations "${ABLATIONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --gpus "${GPUS[@]}" \
    --eta_interval "$ETA_INTERVAL_SEC" \
    --eta_seed_hours "$ETA_SEED_HOURS" \
    "${EXTRA_SWEEP_ARGS[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[run] sweep.py exited rc=$RC at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"

if [[ "$RC" -eq 0 ]]; then
  echo "[run] aggregating results -> results/summary.md" | tee -a "$LOG"
  python scripts/summarize.py --write_csv 2>&1 | tee -a "$LOG"
  echo "[run] DONE. See results/summary.md and $LOG" | tee -a "$LOG"
else
  echo "[run] sweep failed; partial results may still be aggregated:" | tee -a "$LOG"
  python scripts/summarize.py --write_csv 2>&1 | tee -a "$LOG" || true
fi

exit $RC

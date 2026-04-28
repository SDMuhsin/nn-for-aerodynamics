#!/usr/bin/env bash
# One-shot environment setup for a fresh machine.
#
#   bash scripts/setup.sh             # creates env/, installs deps, downloads AirfRANS
#   bash scripts/setup.sh --no-data   # skip the 10 GB AirfRANS download
#
# Idempotent: re-running skips work that's already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DOWNLOAD_DATA=1
for arg in "$@"; do
  case "$arg" in
    --no-data) DOWNLOAD_DATA=0 ;;
    -h|--help)
      sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"

# Compute Canada / DRAC has its own wheelhouse and locks pip to --no-index.
# Hand off to the CC-specific path early so we don't waste time on cu118 fetches.
if [[ -d /cvmfs/soft.computecanada.ca ]]; then
  echo "[setup] Compute Canada CVMFS detected — delegating to scripts/setup_cc.sh"
  exec bash "$SCRIPT_DIR/setup_cc.sh" "$@"
fi

echo "[setup] using Python at: $($PY -c 'import sys; print(sys.executable)')"
PY_VERSION="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.10" ]]; then
  echo "[setup] WARNING: paper used Python 3.10. You have $PY_VERSION."
  echo "[setup]          Continuing anyway."
fi

# 1) Create venv
if [[ ! -d env ]]; then
  echo "[setup] creating venv at ./env"
  "$PY" -m venv env
fi
source env/bin/activate
pip install --quiet --upgrade pip wheel setuptools

# 2) Install torch from PyTorch's cu118 index BEFORE LIPS so its torch==2.0.1
#    pin is satisfied without re-downloading.
if ! python -c 'import torch; assert torch.__version__.startswith("2.0.1")' 2>/dev/null; then
  echo "[setup] installing torch 2.0.1+cu118 from pytorch.org"
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
else
  echo "[setup] torch 2.0.1 already installed"
fi

# 3) Install everything else, including PyG geometric extensions
echo "[setup] installing requirements.txt (LIPS + PyG geometric ext + airfrans)"
pip install --no-cache-dir -f https://data.pyg.org/whl/torch-2.0.1+cu118.html -r requirements.txt

# 4) Quick sanity import check (catches missing PyG ext / lips wiring early)
python - <<'PY'
import torch, torch_geometric, torch_scatter, torch_cluster, torch_sparse, pyg_lib
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import AirfRANSDataSet
from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative
import airfrans, dill, tensorboard
d = Data(pos=torch.randn(100, 2, device="cuda" if torch.cuda.is_available() else "cpu"))
RadiusGraph(r=0.3)(d)
print(f"[setup] OK  torch={torch.__version__}  pyg={torch_geometric.__version__}  "
      f"cuda_available={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
PY

# 5) Download AirfRANS dataset (~10 GB compressed, ~17 GB unpacked)
DATA_DIR="$ROOT/data/airfrans/Dataset"
if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  if [[ -f "$DATA_DIR/manifest.json" ]]; then
    echo "[setup] AirfRANS already present at $DATA_DIR (skipping download)"
  else
    echo "[setup] downloading AirfRANS to $DATA_DIR — this is ~10 GB and takes a while"
    PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python -c "from geompnn.data import download_if_needed; download_if_needed()"
  fi
else
  echo "[setup] --no-data: skipping AirfRANS download"
fi

echo
echo "[setup] DONE."
echo "[setup] Next: bash scripts/run_full_sweep.sh"

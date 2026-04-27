#!/usr/bin/env bash
# Compute Canada / Digital Research Alliance Canada specific setup.
#
# CC's pip is locked to its CVMFS wheelhouse (--no-index by default), and that
# wheelhouse does NOT carry torch 2.0.1 (the paper's version). Lowest +computecanada
# build is 2.1.1, which we accept as a one-minor drift well within the paper's
# ±2 score-point reproduction tolerance.
#
# Usage:
#   module load python/3.10 cuda/11.8
#   bash scripts/setup_cc.sh             # build env + download AirfRANS
#   bash scripts/setup_cc.sh --no-data   # skip download
#
# What it does:
# 1. Installs torch 2.1.1+computecanada and matching PyG ext from the wheelhouse
#    (--no-index path, fast)
# 2. Installs lips-benchmark + airfrans + dill from PyPI via --index-url
#    (CC blocks the default index but allows explicit --index-url)
# 3. Skips the [recommended] extra of lips-benchmark — its torch==2.0.1 hard pin
#    would conflict with the wheelhouse torch, and the AirfRANS scoring path
#    doesn't actually need any of the [recommended] deps (those are for the
#    power-grid use case).
# 4. Verifies imports + a CUDA roundtrip.
# 5. Downloads AirfRANS (~10 GB compressed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DOWNLOAD_DATA=1
for arg in "$@"; do
  case "$arg" in
    --no-data) DOWNLOAD_DATA=0 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

if [[ ! -d /cvmfs/soft.computecanada.ca ]]; then
  echo "[setup_cc] WARNING: this script is tuned for Compute Canada's CVMFS layout."
  echo "[setup_cc]          You're not on CC. Run scripts/setup.sh instead."
  echo "[setup_cc]          Continuing anyway in case you know what you're doing."
fi

echo "[setup_cc] python: $(which python3) ($(python3 -V))"

# 1) Build the venv if missing (don't recreate if user already did `python3 -m venv env`)
if [[ ! -d env ]]; then
  echo "[setup_cc] creating venv at ./env"
  python3 -m venv env
fi
source env/bin/activate

# CC docs recommend --no-download for venv creation; we use the standard one
# but make sure pip itself is current via the wheelhouse.
pip install --no-index --upgrade pip

# 2) Torch + PyG extensions from the CC wheelhouse (--no-index path is fast)
echo "[setup_cc] installing torch 2.1.1 from CC wheelhouse"
pip install --no-index torch==2.1.1

# CC may not pin all PyG ext at exactly the same version — let pip pick what
# it has. PyG 2.x is API-compatible across these patch versions.
echo "[setup_cc] installing torch_geometric + ext from CC wheelhouse"
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric pyg_lib

# 3) Common scientific stack from wheelhouse (LIPS needs these)
echo "[setup_cc] installing scientific stack from CC wheelhouse"
pip install --no-index numpy scipy scikit-learn matplotlib numba pandas pyyaml six pathlib tqdm

# 4) The two libs that aren't in the CC wheelhouse — fetch from PyPI explicitly.
#    --index-url overrides the wheelhouse's default --no-index policy.
echo "[setup_cc] installing lips-benchmark + airfrans + dill from PyPI"
pip install --index-url https://pypi.org/simple --no-deps \
  lips-benchmark==0.2.7 \
  airfrans==0.1.5.1 \
  dill==0.4.1

# tensorboard for logging (CC ships it)
pip install --no-index tensorboard || \
  pip install --index-url https://pypi.org/simple tensorboard

# 5) Sanity import check (catches missing deps before training launches)
python - <<'PY'
import torch, torch_geometric, torch_scatter, torch_cluster, torch_sparse, pyg_lib
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
print(f"[setup_cc] torch={torch.__version__}  pyg={torch_geometric.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import AirfRANSDataSet
from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative
import airfrans, dill, tensorboard
print("[setup_cc] LIPS + airfrans imports OK")
if torch.cuda.is_available():
    d = Data(pos=torch.randn(100, 2, device='cuda'))
    out = RadiusGraph(r=0.3)(d)
    print(f"[setup_cc] PyG RadiusGraph on CUDA: {out.edge_index.shape[1]} edges")
PY

# 6) AirfRANS dataset
DATA_DIR="$ROOT/data/airfrans/Dataset"
if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  if [[ -f "$DATA_DIR/manifest.json" ]]; then
    echo "[setup_cc] AirfRANS already at $DATA_DIR (skipping download)"
  else
    echo "[setup_cc] downloading AirfRANS (~10 GB compressed) to $DATA_DIR"
    PYTHONPATH="$ROOT/src" python -c "from geompnn.data import download_if_needed; download_if_needed()"
  fi
else
  echo "[setup_cc] --no-data: skipping AirfRANS download"
fi

echo
echo "[setup_cc] DONE."
echo "[setup_cc] Next: bash scripts/run_full_sweep.sh"

#!/usr/bin/env bash
# Compute Canada / Digital Research Alliance Canada specific setup.
#
# CC's pip is locked to its CVMFS wheelhouse (--no-index by default), and that
# wheelhouse does NOT carry torch 2.0.1 (the paper's version). Lowest +computecanada
# build is 2.1.1, which we accept as a one-minor drift well within the paper's
# ±2 score-point reproduction tolerance.
#
# We deliberately SKIP pyg_lib / torch_sparse / torch_cluster / torch_scatter /
# torch_spline_conv from the wheelhouse: their compiled .so files have
# repeatedly been observed to have ABI mismatches against the +computecanada
# torch build (e.g. `libpyg.so: undefined symbol: _ZN3c1010Dispatcher...`).
#
# Our model doesn't need them:
#   - All configs set `manual_radius=true`, which routes through our pure-PyTorch
#     `RadGr` class instead of torch_cluster's compiled radius_graph.
#   - `torch_geometric.utils.scatter` falls back to native torch.scatter when
#     torch_scatter is absent. (Mild perf hit, identical numerics.)
#   - `torch_sparse` and `pyg_lib` are only used by torch_geometric's optional
#     fast paths (sparse messsage passing). torch_geometric prints a one-time
#     warning and uses dense fallbacks. We don't trip those paths.
#
# Usage:
#   module purge && module load python/3.11 cuda/11.8   # or python/3.10 if available
#   bash scripts/setup_cc.sh             # build env + download AirfRANS
#   bash scripts/setup_cc.sh --no-data   # skip download
#   bash scripts/setup_cc.sh --repair    # uninstall broken PyG ext, keep rest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DOWNLOAD_DATA=1
REPAIR_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-data) DOWNLOAD_DATA=0 ;;
    --repair)  REPAIR_ONLY=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

if [[ ! -d /cvmfs/soft.computecanada.ca ]]; then
  echo "[setup_cc] WARNING: this script is tuned for Compute Canada's CVMFS layout."
  echo "[setup_cc]          You're not on CC. Run scripts/setup.sh instead."
  echo "[setup_cc]          Continuing anyway in case you know what you're doing."
fi

echo "[setup_cc] python: $(which python3) ($(python3 -V))"

# 1) Build the venv if missing
if [[ ! -d env ]]; then
  echo "[setup_cc] creating venv at ./env"
  python3 -m venv env
fi
source env/bin/activate
pip install --no-index --upgrade pip

# Repair mode: remove any pre-installed broken PyG ext and re-run the sanity
# check. Useful if a prior setup_cc.sh attempt left libpyg.so etc. behind.
if [[ "$REPAIR_ONLY" -eq 1 ]]; then
  echo "[setup_cc] --repair: removing optional PyG ext that ABI-mismatches torch"
  pip uninstall -y pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv 2>/dev/null || true
  echo "[setup_cc] re-running sanity check"
fi

if [[ "$REPAIR_ONLY" -eq 0 ]]; then
  # 2) Torch from CC wheelhouse (closest to paper's 2.0.1)
  if ! python -c 'import torch' 2>/dev/null; then
    echo "[setup_cc] installing torch from CC wheelhouse (closest available to 2.0.1)"
    pip install --no-index torch==2.1.1 || pip install --no-index torch
  else
    echo "[setup_cc] torch already installed: $(python -c 'import torch; print(torch.__version__)')"
  fi

  # 3) torch_geometric core (pure-Python, no compiled C++) from CC wheelhouse
  pip install --no-index torch_geometric

  # 4) Scientific stack (LIPS' actual deps for the airfrans path)
  pip install --no-index numpy scipy scikit-learn matplotlib numba pandas pyyaml six tqdm

  # 5) lips-benchmark + airfrans + dill from PyPI
  #    --index-url overrides CC's default --no-index policy. CC's compute nodes
  #    sometimes block outbound PyPI; if that happens, run this script on a
  #    LOGIN node first (which always has PyPI), then re-run on the compute node.
  echo "[setup_cc] installing lips-benchmark + airfrans + dill from PyPI (--index-url override)"
  pip install --index-url https://pypi.org/simple --no-deps \
    lips-benchmark==0.2.7 \
    airfrans==0.1.5.1 \
    dill==0.4.1

  # 6) tensorboard (try wheelhouse first, fall back to PyPI)
  pip install --no-index tensorboard 2>/dev/null || \
    pip install --index-url https://pypi.org/simple tensorboard
fi

# 7) Sanity check — strict on torch + torch_geometric + lips + airfrans, lenient
#    on the optional PyG extension family.
python - <<'PY'
import importlib, sys, traceback
ok = True

def must(pkg):
    global ok
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "<no __version__>")
        print(f"[sanity] OK   {pkg:25s} version={ver}")
    except Exception as e:
        ok = False
        print(f"[sanity] FAIL {pkg:25s} :: {type(e).__name__}: {e}")

def optional(pkg):
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "<no __version__>")
        print(f"[sanity] opt+ {pkg:25s} version={ver}")
    except Exception as e:
        print(f"[sanity] opt- {pkg:25s} (skipped: {type(e).__name__}: {str(e)[:80]})")

must("torch")
must("torch_geometric")
must("numpy")
must("scipy")
must("scipy.stats")
must("sklearn")
must("airfrans")
must("dill")
must("tensorboard")

# These are the four LIPS modules our simulator actually touches
must("lips.benchmark.airfransBenchmark")
must("lips.dataset.airfransDataSet")
must("lips.dataset.scaler.standard_scaler_iterative")
must("lips.evaluation.airfrans_evaluation")

# Optional extensions — if they crash, our manual_radius path handles it
optional("torch_scatter")
optional("torch_cluster")
optional("torch_sparse")
optional("torch_spline_conv")
optional("pyg_lib")

# Real smoke: build the model on whatever device is available. This catches
# any remaining ABI / import / config issue before training launches.
import torch
print()
print(f"[sanity] CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
sys.path.insert(0, "src")
from geompnn.simulator import GNN
hparams = dict(
    hidden_dim=64, target_dim=1, num_layers=4,
    vol_radius=0.0, vol_max_num_neighbors=8,
    surf_radius=0.05, surf_max_num_neighbors=8,
    surf_spacing=1, s2v_max_num_neighbors=8,
    manual_radius=True,
    ang_basis=8, dist_basis=8, coord_basis=8,
    coord_spacing=0.001, coord_max=8,
    angles="null", dist="identity", coords="identity",
    canonical_coords=True, inlet_coords=False, tail_coords=False,
    closest_coords=False, norm_type="mlp_post", edge_norm=False,
)
m = GNN(hparams=hparams, field="x-velocity")
n_params = sum(p.numel() for p in m.parameters())
print(f"[sanity] OK   built S2V GNN model, {n_params:,} params")
print()
if not ok:
    print("[sanity] FAILURES above — install is incomplete.")
    sys.exit(1)
print("[sanity] all required imports + model construction OK")
PY

# 8) AirfRANS dataset
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

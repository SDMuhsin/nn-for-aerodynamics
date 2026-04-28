#!/usr/bin/env bash
# Compute Canada / Digital Research Alliance Canada specific setup.
#
# WHEELHOUSE LIMITATION (root cause of vtk install failure, diagnosed
# 2026-04-28 via scripts/diagnose_cc_vtk*.sh):
#   - CC's pip 26.0.1+computecanada lists 39 compatible platform tags but
#     EXCLUDES every manylinux variant. PyPI ships every cp311+linux vtk wheel
#     ONLY as manylinux_2_17_x86_64 — so pip and PyPI cannot intersect for vtk
#     on Python 3.11. The CC wheelhouse itself only has cp310 vtk wheels.
#   - The fix: take vtk from the system module (`module load vtk/9.3.0`).
#     The bindings live at $EBROOTVTK/lib/python3.11/site-packages and are
#     auto-added to the venv's sys.path via CC's $EBPYTHONPREFIXES sitecustomize.
#   - The module load chain `python/3.11 vtk/9.3.0` REQUIRES `set -u` to be
#     OFF: Lmod's bash function silently aborts under strict-unset mode
#     (round-2 evidence). Hence this script uses only `-eo pipefail`.
#
# We also deliberately SKIP pyg_lib / torch_sparse / torch_cluster /
# torch_scatter / torch_spline_conv: their compiled .so files have ABI
# mismatches against the +computecanada torch build. Our model doesn't need
# them (manual_radius=true configs route around torch_cluster.radius_graph;
# torch_geometric.utils.scatter falls back to native torch.scatter).
#
# Torch is taken from the wheelhouse at 2.1.1+computecanada (paper used 2.0.1;
# one minor drift, within the paper's ±2 score-point tolerance).
#
# Usage:
#   bash scripts/setup_cc.sh             # build env + download AirfRANS
#   bash scripts/setup_cc.sh --no-data   # skip download
#   bash scripts/setup_cc.sh --repair    # uninstall broken PyG ext, keep rest

# NOTE: NO `set -u`. CC's Lmod `module` bash function references vars that
# are unset on first invocation; `set -u` aborts it silently. See diagnostic.
set -eo pipefail

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

# Load the system modules. vtk is the critical one: we cannot pip-install it
# (see header comment), so we use CC's system build via Lmod, which exposes the
# Python bindings via $EBPYTHONPREFIXES. python/3.11 is required by vtk/9.3.0
# (per `module spider vtk/9.3.0`). StdEnv/2023 + gcc/12.3 are usually already
# loaded from login but we re-load to be defensive.
echo "[setup_cc] loading system modules: StdEnv/2023 gcc/12.3 python/3.11 vtk/9.3.0"
module load StdEnv/2023 gcc/12.3 python/3.11 vtk/9.3.0
echo "[setup_cc] EBROOTVTK=${EBROOTVTK:-NOT-SET}  EBVERSIONVTK=${EBVERSIONVTK:-NOT-SET}"
if [[ -z "${EBROOTVTK:-}" ]]; then
  echo "[setup_cc] FATAL: vtk module did not load. EBROOTVTK is unset."
  echo "[setup_cc]        Try 'module spider vtk/9.3.0' to see prerequisites."
  exit 5
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

  # 4b) airfrans loads each simulation's .vtu/.vtp via pyvista (which wraps VTK).
  #     We cannot pip-install vtk: PyPI ships only manylinux-tagged cp311 vtk
  #     wheels and CC's pip excludes manylinux from its compatible-tag list.
  #     Instead we take vtk from the system module (loaded at script start),
  #     which sets EBPYTHONPREFIXES so the venv's python finds vtk transparently.
  #
  #     Pyvista has runtime deps beyond vtk (scooby, pooch, pillow, ...). To
  #     skip vtk while still installing the rest, we install pyvista in two
  #     steps: (i) install everything pyvista needs EXCEPT vtk, (ii) install
  #     pyvista itself with --no-deps. The dep list comes from the actual
  #     wheel's METADATA so it stays correct if pyvista's deps change.
  PYVISTA_WHL=$(ls /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic/pyvista-*.whl 2>/dev/null | sort -V | tail -1)
  if [[ -z "$PYVISTA_WHL" ]]; then
    echo "[setup_cc] FATAL: no pyvista wheel in CC wheelhouse"
    exit 6
  fi
  echo "[setup_cc] using pyvista wheel: $(basename "$PYVISTA_WHL")"

  PYVISTA_DEPS=$(unzip -p "$PYVISTA_WHL" '*/METADATA' | awk '
    /^Requires-Dist:/ {
      sub(/^Requires-Dist: */, "")
      if ($0 ~ /^vtk([^A-Za-z0-9_]|$)/) next   # vtk comes from system module
      if ($0 ~ /extra ==/)               next  # optional extras (jupyter, ...)
      print
    }')
  echo "[setup_cc] pyvista runtime deps (excluding vtk + extras):"
  echo "$PYVISTA_DEPS" | sed 's/^/  /'
  if [[ -n "$PYVISTA_DEPS" ]]; then
    # xargs splits on newlines so each spec like 'scooby>=0.5.1' stays one arg.
    printf '%s\n' "$PYVISTA_DEPS" | xargs -r -d '\n' pip install --no-index
  fi

  echo "[setup_cc] verifying vtk is importable from the system module"
  python -c "import vtk; v = vtk.vtkVersion.GetVTKVersion(); print(f'[setup_cc]   vtk {v}'); assert v.startswith('9.3'), f'expected 9.3.x, got {v}'"
  echo "[setup_cc] installing pyvista from the wheel itself (--no-deps; vtk module-provided)"
  pip install --no-index --no-deps "$PYVISTA_WHL"

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
    # PREPEND to PYTHONPATH — don't overwrite. CC's Python relies on a
    # sitecustomize.py at /cvmfs/.../easybuild/python/site-packages/ that
    # walks $EBPYTHONPREFIXES to add the system vtk's path to sys.path.
    # Replacing PYTHONPATH strips that directory, sitecustomize never runs,
    # and `import vtkmodules` fails.
    PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python -c "from geompnn.data import download_if_needed; download_if_needed()"
  fi
else
  echo "[setup_cc] --no-data: skipping AirfRANS download"
fi

echo
echo "[setup_cc] DONE."
echo "[setup_cc] Next: bash scripts/run_full_sweep.sh"

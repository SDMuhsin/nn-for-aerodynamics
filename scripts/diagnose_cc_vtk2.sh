#!/usr/bin/env bash
# diagnose_cc_vtk2.sh — round 2 of vtk root-cause probing.
#
# Findings from round 1:
#   - constraints.txt does NOT pin vtk → ruled out.
#   - pip config does NOT have index-url / no-index → ruled out as a CONFIG cause.
#   - HOWEVER, every dry-run probe (even with --isolated --index-url PyPI) failed
#     with "from versions: none" and pip never printed "Looking in indexes:".
#     That means pip is somehow not querying any index, despite --index-url
#     being passed explicitly. We have not yet found WHERE this is enforced.
#
# Three remaining hypotheses to test:
#   H1: There is another pip config file we haven't read. The verbose config
#       list mentions /cvmfs/.../gentoo/2023/x86-64-v3/etc/xdg/pip/pip.conf as
#       a 'global' candidate — never inspected.
#   H2: The pip in the venv (pip 26.0.1) is CC-patched to default to --no-index.
#       pip 26 is suspicious; current upstream pip is 24.x.
#   H3: `module load vtk/9.3.0` failed silently because of a missing
#       prerequisite (probably python/3.11). Need module spider to confirm.
#
# Usage (same pattern as round 1):
#   source env/bin/activate
#   bash scripts/diagnose_cc_vtk2.sh > /tmp/diagnose_cc_vtk2.log 2>&1
#   cat /tmp/diagnose_cc_vtk2.log

set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

section() { printf '\n========== %s ==========\n' "$1"; }

section "0. environment recap"
echo "host:   $(hostname)"
echo "venv:   ${VIRTUAL_ENV:-NOT-ACTIVATED}"
echo "python: $(command -v python) ($(python -V 2>&1))"
echo "pip:    $(command -v pip) ($(pip --version 2>&1))"

# ============================================================================
# H1: Are there additional pip config files we missed?
# ============================================================================
section "H1.1 — every potential pip config path: existence + contents"
for f in \
  /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/etc/xdg/pip/pip.conf \
  /etc/xdg/pip/pip.conf \
  /etc/pip.conf \
  "${VIRTUAL_ENV:-/nonexistent}/pip.conf" \
  /cvmfs/soft.computecanada.ca/config/python/pip-x86-64-v3-gentoo2023.conf
do
  if [[ -f "$f" ]]; then
    echo "EXISTS  $f"
    echo "--- contents ---"
    cat "$f"
    echo "---"
  else
    echo "MISSING $f"
  fi
done

section "H1.2 — pip config debug + provenance (looking for hidden index disablement)"
pip config debug 2>&1 | head -80 || pip config list -v 2>&1 | head -80

# ============================================================================
# H2: Is the venv's pip CC-patched?
# ============================================================================
section "H2.1 — pip module location + version metadata"
python -c "import pip, os; print('pip __version__:', pip.__version__); print('pip __file__:', pip.__file__); print('pip dir:', os.path.dirname(pip.__file__))"
echo
echo "--- pip's package METADATA (top of file) ---"
pip show pip 2>&1 | head -20
echo
echo "--- list of files in pip's installation, looking for CC-specific patches ---"
PIP_DIR=$(python -c "import pip, os; print(os.path.dirname(pip.__file__))" 2>/dev/null)
if [[ -d "$PIP_DIR" ]]; then
  echo "pip installed at: $PIP_DIR"
  ls "$PIP_DIR" | head -10
  echo "--- any reference to 'computecanada' or 'cvmfs' inside pip's source? ---"
  grep -rIl --include='*.py' -E 'computecanada|/cvmfs/|--no-index' "$PIP_DIR" 2>/dev/null | head -10
fi

section "H2.2 — what does 'pip install --dry-run -vv' actually do? (full network trace)"
echo "Running with -vv (very verbose) to see HTTP traffic. Truncated to 120 lines."
pip install --dry-run -vv --index-url https://pypi.org/simple "vtk==9.3.1" 2>&1 | head -120
echo "(exit code: ${PIPESTATUS[0]})"

section "H2.3 — try with PIP_CONFIG_FILE explicitly unset (env-level reset)"
echo "cmd: env -u PIP_CONFIG_FILE pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple --no-build-isolation \"vtk==9.3.1\""
env -u PIP_CONFIG_FILE pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple --no-build-isolation "vtk==9.3.1" 2>&1 | head -40
echo "(exit code: ${PIPESTATUS[0]})"

section "H2.4 — try with PIP_CONFIG_FILE=/dev/null (force empty config)"
echo "cmd: PIP_CONFIG_FILE=/dev/null pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple \"vtk==9.3.1\""
PIP_CONFIG_FILE=/dev/null pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple "vtk==9.3.1" 2>&1 | head -40
echo "(exit code: ${PIPESTATUS[0]})"

section "H2.5 — direct PyPI HTTP fetch (does pip's transport layer work at all?)"
echo "--- curl pypi.org/simple/vtk/ ---"
curl -s -o /tmp/_vtk_simple.html -w 'http_code=%{http_code} size=%{size_download} time=%{time_total}\n' --max-time 15 https://pypi.org/simple/vtk/
echo "first cp311 wheel referenced in PyPI's vtk index page:"
grep -oE 'vtk-9\.[0-9]+\.[0-9]+-cp311-[^"]+\.whl' /tmp/_vtk_simple.html 2>/dev/null | head -3 || echo "(no cp311 hits via curl — but they exist on PyPI)"
echo
echo "--- pip's resolver against PyPI directly via Python (bypass pip's CLI defaults) ---"
python - <<'PY' 2>&1 | head -30
import urllib.request, ssl, sys
url = 'https://pypi.org/simple/vtk/'
try:
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.pypi.simple.v1+json'})
    with urllib.request.urlopen(req, timeout=15, context=ssl.create_default_context()) as r:
        body = r.read().decode()
    cp311_hits = [ln for ln in body.split('\n') if 'cp311' in ln and 'vtk-9.3' in ln][:3]
    print(f'GOT {len(body)} bytes from {url}')
    print(f'cp311 vtk 9.3.x mentions in response:')
    for h in cp311_hits: print('  ', h.strip()[:200])
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
PY

# ============================================================================
# H3: Module load — does vtk/9.3.0 need a prerequisite module?
# ============================================================================
section "H3.1 — module spider vtk/9.3.0 (lists prerequisites)"
module spider vtk/9.3.0 2>&1 | head -50

section "H3.2 — try the documented load sequence (with python/3.11 first)"
echo "--- before: PYTHONPATH and which python ---"
echo "PYTHONPATH=${PYTHONPATH:-(unset)}"
echo "python   = $(command -v python)"
echo
echo "--- module load python/3.11 ---"
module load python/3.11 2>&1 | head -10
echo
echo "--- module load vtk/9.3.0 (after python/3.11) ---"
module load vtk/9.3.0 2>&1 | head -10
echo
echo "--- module list (post-load) ---"
module list 2>&1
echo
echo "--- PYTHONPATH after both loads ---"
echo "${PYTHONPATH:-(unset)}"
echo
echo "--- VTK / EBROOT / EBVERSION env vars ---"
env | grep -iE '^(EBROOT|EBVERSION|VTK_|PYTHONHOME)' | head -10 || echo "(none)"
echo
echo "--- can the venv python now import vtk? ---"
# IMPORTANT: still call the venv python explicitly to check if PYTHONPATH inheritance works
"$VIRTUAL_ENV/bin/python" -c "import vtk; print('SUCCESS:', vtk.vtkVersion.GetVTKVersion())" 2>&1 | head -5

section "H3.3 — alternative: try the load sequence WITHOUT venv (system python)"
echo "(this checks whether vtk's python bindings work at all when the right python is loaded)"
deactivate 2>/dev/null || true
echo "after deactivate: which python = $(command -v python 2>/dev/null) ($(python -V 2>&1 || true))"
python -c "import vtk; print('SUCCESS:', vtk.vtkVersion.GetVTKVersion())" 2>&1 | head -5

# clean up modules
module unload vtk python 2>/dev/null || true

section "DONE — paste full log back"

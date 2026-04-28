#!/usr/bin/env bash
# diagnose_cc_vtk3.sh — round 3, going for the kill.
#
# Round 2 nailed two big findings:
#   (a) pip IS hitting PyPI; rejection is per-wheel tag matching, NOT no-index.
#   (b) `module load` silently fails inside our bash script (no env change after).
#
# Three remaining things to nail down:
#   F1: What tags does CC's pip 26.0.1+computecanada actually consider
#       compatible? (`pip debug --verbose`) — settles whether cp311 manylinux
#       wheels CAN ever be installed via pip on this system.
#   F2: If we manually download a cp311 vtk wheel from PyPI, does pip install
#       it from a local file? — bypasses index resolution but still subject to
#       tag matching.
#   F3: Why is `module load` silently no-op'ing? The candidates are: (i) `set
#       -u`/`set -e` interacting with Lmod's bash function which references
#       maybe-unset vars, (ii) Lmod's function not exported into the script's
#       subshell, (iii) the venv-activated shell has had Lmod's init torn out.
#       We probe each.
#
#   IMPORTANT: this script DELIBERATELY does NOT use `set -u` or `set -e` —
#   that was the suspected cause of round 2's silent module-load failure.
#
# Usage:
#   source env/bin/activate
#   bash scripts/diagnose_cc_vtk3.sh > /tmp/diagnose_cc_vtk3.log 2>&1
#   cat /tmp/diagnose_cc_vtk3.log

# Note: NO set -u, NO set -e — see comment above.
set -o pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

section() { printf '\n========== %s ==========\n' "$1"; }

section "0. environment"
echo "host:    $(hostname)"
echo "venv:    ${VIRTUAL_ENV:-NOT-ACTIVATED}"
echo "python:  $(command -v python) ($(python -V 2>&1))"
echo "pip:     $(pip --version 2>&1)"

# ============================================================================
# F1. What tags does CC's pip consider compatible? Decisive for "is the wheel
#     even installable from a local file".
# ============================================================================
section "F1. pip debug --verbose (compatible tags + platform)"
pip debug --verbose 2>&1 | head -80

# Reference: what tags does PyPI's vtk 9.3.1 actually ship for cp311 linux?
section "F1b. PyPI: which vtk-9.3.x wheels exist for cp311? (pulled via curl, not pip)"
curl -s --max-time 20 https://pypi.org/simple/vtk/ -o /tmp/_pypi_vtk.html
echo "page size: $(wc -c < /tmp/_pypi_vtk.html) bytes"
echo
echo "--- all cp311 wheel filenames (any version) ---"
grep -oE 'vtk-[0-9.]+-cp311-cp311-[^"<]+\.whl' /tmp/_pypi_vtk.html | sort -u
echo
echo "--- specifically cp311 + linux (manylinux/musllinux) ---"
grep -oE 'vtk-[0-9.]+-cp311-cp311-(manylinux|musllinux)[^"<]+\.whl' /tmp/_pypi_vtk.html | sort -u

# ============================================================================
# F2. If we manually fetch the wheel from PyPI, does pip install it locally?
#     This bypasses --index-url resolution; tag matching is the only filter.
# ============================================================================
section "F2. download vtk-9.3.1 cp311 manylinux wheel from PyPI directly"
WHEEL_URL=$(grep -oE 'https://files\.pythonhosted\.org/[^"<]*vtk-9\.3\.[0-9]+-cp311-cp311-manylinux[^"<]*\.whl' /tmp/_pypi_vtk.html | head -1)
echo "wheel URL: ${WHEEL_URL:-NONE FOUND}"
if [[ -n "$WHEEL_URL" ]]; then
  rm -f /tmp/vtk_local.whl
  curl -sL --max-time 120 -o /tmp/vtk_local.whl "$WHEEL_URL"
  ls -la /tmp/vtk_local.whl
  echo
  section "F2a. inspect local wheel's metadata (tags it claims)"
  python -c "
from packaging.utils import parse_wheel_filename
import os
fn = os.path.basename('/tmp/vtk_local.whl')
# Reconstruct original filename from wheel metadata (we renamed it on download)
url='$WHEEL_URL'
orig = url.rsplit('/',1)[1]
print('downloaded as:', orig)
n,v,b,tags = parse_wheel_filename(orig)
print('name:', n, 'version:', v, 'tags:', tags)
"
  echo
  section "F2b. pip install --dry-run from the local wheel file"
  echo "cmd: pip install --dry-run --no-deps --no-cache-dir /tmp/vtk_local.whl"
  pip install --dry-run --no-deps --no-cache-dir /tmp/vtk_local.whl 2>&1 | head -30
  echo "(exit code: ${PIPESTATUS[0]})"
  echo
  section "F2c. pip install --dry-run --target= (alternate path; may avoid some tag enforcement)"
  rm -rf /tmp/vtk_target_probe
  echo "cmd: pip install --dry-run --no-deps --target=/tmp/vtk_target_probe /tmp/vtk_local.whl"
  pip install --dry-run --no-deps --target=/tmp/vtk_target_probe /tmp/vtk_local.whl 2>&1 | head -30
  echo "(exit code: ${PIPESTATUS[0]})"
fi

# ============================================================================
# F3. Why is module load silently failing? Try multiple environments.
# ============================================================================
section "F3.1 — is the module function defined? Where does it come from?"
type module 2>&1 | head -10
echo
echo "--- LMOD env vars ---"
env | grep -E '^(LMOD|MODULE|_LMFILES_|MODULESHOME|BASH_ENV)' | head -20

section "F3.2 — try module load WITHOUT set -u, WITHOUT pipefail, capturing exit codes"
# Save current modules so we can detect any change
BEFORE=$(module list 2>&1)
echo "--- before: --- "
echo "$BEFORE" | head -20
echo

# Method 1: direct call (current method; failed in round 2)
echo "--- METHOD 1: direct 'module load python/3.11' ---"
module load python/3.11
echo "exit code: $?"
module list 2>&1 | head -10
echo

# Method 2: via $LMOD_CMD (lower level)
if [[ -n "${LMOD_CMD:-}" ]]; then
  echo "--- METHOD 2: via \$LMOD_CMD ($LMOD_CMD) ---"
  eval "$($LMOD_CMD bash load python/3.11 2>&1)"
  echo "after eval — module list:"
  module list 2>&1 | head -10
  echo
fi

# Method 3: source the lmod init and retry
echo "--- METHOD 3: source /cvmfs/.../lmod init then try module load ---"
LMOD_INIT_CANDIDATES=(
  /cvmfs/soft.computecanada.ca/custom/init/profile/local-modules.sh
  /cvmfs/soft.computecanada.ca/custom/init/sh/lmod.sh
  /etc/profile.d/lmod.sh
  /etc/profile.d/00-modulepath.sh
)
for f in "${LMOD_INIT_CANDIDATES[@]}"; do
  if [[ -f "$f" ]]; then
    echo "found init file: $f"
    source "$f" 2>&1 | head -5
  fi
done
echo "after sourcing — try 'module load python/3.11':"
module load python/3.11
echo "exit code: $?"
echo "module list now:"
module list 2>&1 | head -10
echo
echo "PYTHONPATH after: ${PYTHONPATH:-(unset)}"
echo

# Method 4: load the full chain explicitly
echo "--- METHOD 4: load python/3.11 then vtk/9.3.0 (full chain) ---"
module load python/3.11 vtk/9.3.0
echo "exit code: $?"
echo "module list:"
module list 2>&1 | head -10
echo
echo "PYTHONPATH: ${PYTHONPATH:-(unset)}"
echo "LD_LIBRARY_PATH (vtk-related entries):"
echo "${LD_LIBRARY_PATH:-(unset)}" | tr ':' '\n' | grep -i vtk || echo "(no vtk in LD_LIBRARY_PATH)"
echo
echo "EBROOTVTK / EBVERSIONVTK:"
env | grep -iE 'EBROOTVTK|EBVERSIONVTK' || echo "(none)"
echo
echo "--- the moment of truth: can python import vtk? ---"
"$VIRTUAL_ENV/bin/python" -c "import sys; print('python:', sys.executable, sys.version); import vtk; print('SUCCESS: vtk', vtk.vtkVersion.GetVTKVersion())" 2>&1 | head -10

# ============================================================================
# F4. Sanity: if F3 method 4 succeeded, what env vars does the venv-python
#     inherit? Capture the full picture for setup_cc.sh.
# ============================================================================
section "F4. capture full env (relevant subset) after the working module load"
echo "--- PATH ---"
echo "$PATH" | tr ':' '\n' | head -10
echo
echo "--- PYTHONPATH ---"
echo "${PYTHONPATH:-(unset)}" | tr ':' '\n' | head -10
echo
echo "--- LD_LIBRARY_PATH (top entries) ---"
echo "${LD_LIBRARY_PATH:-(unset)}" | tr ':' '\n' | head -15
echo
echo "--- all VTK / EBROOT / EBVERSION env vars ---"
env | grep -iE 'VTK|EBROOT|EBVERSION' | head -30
echo
echo "--- where will python find vtk? ---"
"$VIRTUAL_ENV/bin/python" -c "
import sys
for p in sys.path:
    try:
        import os
        if os.path.exists(os.path.join(p, 'vtk.py')) or os.path.exists(os.path.join(p, 'vtkmodules')):
            print('FOUND vtk in:', p)
    except Exception: pass
" 2>&1 | head -10

section "DONE"

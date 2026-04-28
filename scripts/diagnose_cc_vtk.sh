#!/usr/bin/env bash
# diagnose_cc_vtk.sh — root-cause probe for "vtk install fails on CC" specifically.
#
# What this script does NOT do: it does not install anything. It uses pip's
# --dry-run flag (which resolves and fetches metadata but does not install) and
# `module load` (which is shell-local and does not persist after the script).
#
# Goal: figure out, from evidence rather than hypothesis, why we cannot get a
# working `vtk` Python module on this Compute Canada Python 3.11 venv.
#
# Run from project root, with the venv activated:
#   source env/bin/activate
#   bash scripts/diagnose_cc_vtk.sh > /tmp/diagnose_cc_vtk.log 2>&1
#   cat /tmp/diagnose_cc_vtk.log
# Then paste the full log back.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

section() {
  printf '\n========== %s ==========\n' "$1"
}

# ============================================================================
# 0. Sanity: are we in the right shell? Captured for the audit trail.
# ============================================================================
section "host & python (sanity)"
echo "host:        $(hostname)"
echo "user:        $(whoami)"
echo "date:        $(date)"
echo "pwd:         $(pwd)"
echo "venv:        ${VIRTUAL_ENV:-NOT-ACTIVATED}"
echo "python:      $(command -v python || echo MISSING)  ($(python -V 2>&1 || true))"
echo "pip:         $(command -v pip || echo MISSING)  ($(pip --version 2>&1 || true))"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo
  echo "WARNING: venv not activated. Most probes below will report system pip,"
  echo "which is not what we want. Activate first:  source env/bin/activate"
fi

# ============================================================================
# 1. The constraints.txt file referenced by pip config — never captured before.
# ============================================================================
section "1. CONSTRAINTS FILE referenced by pip config (the most likely culprit)"
CONSTRAINTS_FILE="/cvmfs/soft.computecanada.ca/config/python/constraints.txt"
if [[ -f "$CONSTRAINTS_FILE" ]]; then
  echo "path:  $CONSTRAINTS_FILE"
  echo "size:  $(wc -l < "$CONSTRAINTS_FILE") lines, $(wc -c < "$CONSTRAINTS_FILE") bytes"
  echo
  echo "--- lines mentioning vtk ---"
  grep -i 'vtk' "$CONSTRAINTS_FILE" || echo "(none)"
  echo
  echo "--- lines mentioning pyvista ---"
  grep -i 'pyvista' "$CONSTRAINTS_FILE" || echo "(none)"
  echo
  echo "--- first 40 lines (context) ---"
  head -40 "$CONSTRAINTS_FILE"
else
  echo "MISSING: $CONSTRAINTS_FILE"
fi

# ============================================================================
# 2. The pip config file itself — re-read with full provenance.
# ============================================================================
section "2. pip config file contents (raw)"
PIP_CONF="/cvmfs/soft.computecanada.ca/config/python/pip-x86-64-v3-gentoo2023.conf"
if [[ -f "$PIP_CONF" ]]; then
  echo "path: $PIP_CONF"
  echo "--- full file ---"
  cat "$PIP_CONF"
else
  echo "MISSING: $PIP_CONF"
fi

section "3. pip config list -v (verbose: shows precedence + which file each came from)"
pip config list -v 2>&1 | head -60

section "4. all PIP_* env vars (any of these could secretly override pip flags)"
env | grep -E '^PIP_' || echo "(none)"

# ============================================================================
# 3. Behaviour probes: what does pip ACTUALLY do for vtk under each flag combo?
#    --dry-run resolves and reports without installing; safe to run repeatedly.
# ============================================================================
echo
echo "============================================================"
echo "  PIP DRY-RUN PROBES"
echo "  Each probe asks pip 'what would you do if I asked you to"
echo "  install vtk under THESE flags?', and reports the answer."
echo "  Nothing is actually installed."
echo "============================================================"

probe() {
  local label="$1"; shift
  section "PROBE: $label"
  printf 'cmd: pip install --dry-run %s "vtk>=9.3,<9.4"\n' "$*"
  echo "--- output ---"
  pip install --dry-run "$@" "vtk>=9.3,<9.4" 2>&1 | head -50
  local rc=${PIPESTATUS[0]}
  echo "--- exit code: $rc ---"
}

probe "A: defaults (with CC pip config + constraints active)"
probe "B: --isolated (ignore CC config files + env vars)"                     --isolated
probe "C: --isolated + explicit PyPI index"                                   --isolated --index-url https://pypi.org/simple
probe "D: --isolated + PyPI + --no-deps (skip transitive resolution)"         --isolated --no-deps --index-url https://pypi.org/simple
probe "E: --no-build-isolation + PyPI (explore alternate cause)"              --no-build-isolation --index-url https://pypi.org/simple

# pip download is occasionally less constrained than `pip install`.
section "PROBE F: 'pip download' (downloads to /tmp; doesn't install)"
echo "cmd: pip download --no-deps --isolated --index-url https://pypi.org/simple --dest /tmp/vtk_probe \"vtk>=9.3,<9.4\""
mkdir -p /tmp/vtk_probe
echo "--- output ---"
pip download --no-deps --isolated --index-url https://pypi.org/simple --dest /tmp/vtk_probe "vtk>=9.3,<9.4" 2>&1 | head -50
echo "--- exit code: ${PIPESTATUS[0]} ---"
echo "--- contents of /tmp/vtk_probe after probe ---"
ls -la /tmp/vtk_probe 2>&1 | head -10

# ============================================================================
# 4. Alternative path: the system VTK module.
#    If 'module load vtk/9.3.0' provides Python bindings on PYTHONPATH, we may
#    not need pip-vtk at all — pyvista from the wheelhouse can use system vtk.
# ============================================================================
section "5. SYSTEM VTK MODULE — does 'module load vtk/9.3.0' provide a Python-importable vtk?"
# Save initial state for diff
echo "--- modules currently loaded ---"
module list 2>&1 | tail -8
echo
echo "--- attempting: module load vtk/9.3.0 ---"
module load vtk/9.3.0 2>&1 | head -10
echo
echo "--- modules after load ---"
module list 2>&1 | tail -8
echo
echo "--- VTK-related env vars after load ---"
env | grep -iE '^(VTK|EBROOT|EBVERSION).*VTK' | head -10 || echo "(none)"
echo
echo "--- PYTHONPATH after load ---"
echo "${PYTHONPATH:-(unset)}"
echo
echo "--- python sys.path after load (relevant entries) ---"
python -c "import sys; [print('  '+p) for p in sys.path]" 2>&1 | head -20
echo
echo "--- can Python import vtk now? ---"
python -c "import vtk; print('SUCCESS: vtk', vtk.vtkVersion.GetVTKVersion())" 2>&1 | head -10
echo
echo "--- where do vtk python files actually live (if anywhere)? ---"
find /cvmfs -path '*/vtk/9.3.0/*' -name 'vtk.py' 2>/dev/null | head -5
find /cvmfs -path '*/vtk*/9.3*' -name 'vtkmodules' -type d 2>/dev/null | head -5
echo
echo "--- attempting: module load vtk/9.4.2 (default) ---"
module unload vtk 2>/dev/null || true
module load vtk/9.4.2 2>&1 | head -10
echo "--- python import vtk after 9.4.2 load ---"
python -c "import vtk; print('SUCCESS: vtk', vtk.vtkVersion.GetVTKVersion())" 2>&1 | head -5

# Restore by unloading the test module (modules are shell-local; this is just hygiene)
module unload vtk 2>/dev/null || true

# ============================================================================
# 5. Wheelhouse pyvista probe: which python tags do the wheelhouse pyvista
#    wheels actually claim, and would they install on cp311?
# ============================================================================
section "6. PYVISTA wheelhouse tag inspection (what python versions does the wheel claim?)"
WH="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
if [[ -d "$WH" ]]; then
  for whl in "$WH"/pyvista-*.whl; do
    [[ -f "$whl" ]] || continue
    echo "wheel: $(basename "$whl")"
    # The METADATA file inside the .whl declares Python-Version + Requires-Dist
    metadata=$(unzip -p "$whl" '*/METADATA' 2>/dev/null | head -200)
    echo "$metadata" | grep -iE '^(Requires-Python|Requires-Dist:.*vtk)' | head -5
    echo "---"
  done
fi

# ============================================================================
# 6. Summary.
# ============================================================================
section "SUMMARY — what to look at when reading this output"
echo "Q1: Does constraints.txt pin vtk or pyvista?    => see section 1"
echo "Q2: Does the pip config truly disable PyPI?     => see section 2 (look for index-url / no-index)"
echo "Q3: Does --isolated bypass the config?          => compare PROBE A vs B"
echo "Q4: Does PROBE D (the script's intended fix)    => see PROBE D exit code"
echo "    actually succeed?"
echo "Q5: Is 'module load vtk' a viable alternative?  => see section 5"
echo "Q6: Does wheelhouse pyvista require a vtk that  => see section 6 (Requires-Dist:vtk)"
echo "    cp311 cannot satisfy?"
echo
echo "Send the full log of this script back. Do not interpret in-place."

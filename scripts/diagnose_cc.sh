#!/usr/bin/env bash
# Compute Canada install diagnostic. Read-only; produces a single report you
# can paste back so we don't keep guessing at install commands.
#
#   bash scripts/diagnose_cc.sh
#   bash scripts/diagnose_cc.sh --quiet   # skip stderr noise from probes

set -uo pipefail

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1
errsink() { if [[ $QUIET -eq 1 ]]; then "$@" 2>/dev/null; else "$@"; fi; }

section() {
  printf '\n========== %s ==========\n' "$1"
}

section "host & python"
echo "host:        $(hostname)"
echo "user:        $(whoami)"
echo "date:        $(date)"
echo "python3:     $(command -v python3 || echo MISSING)  ($(python3 -V 2>&1 || true))"
if [[ -d env ]]; then
  echo "venv:        ./env  (active=${VIRTUAL_ENV:-no})"
else
  echo "venv:        ./env  (not present)"
fi

section "loaded modules"
errsink module list 2>&1 || echo "module command unavailable"

section "wheelhouse search paths"
WH_PATHS=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)
for p in "${WH_PATHS[@]}"; do
  if [[ -d "$p" ]]; then echo "EXISTS  $p"; else echo "MISSING $p"; fi
done

section "wheelhouse: vtk-related wheels (any name containing 'vtk')"
for p in "${WH_PATHS[@]}"; do
  [[ -d "$p" ]] || continue
  matches=$(ls -1 "$p" 2>/dev/null | grep -i vtk || true)
  if [[ -n "$matches" ]]; then
    echo "--- $p ---"
    echo "$matches" | head -40
  fi
done

section "wheelhouse: pyvista wheels"
for p in "${WH_PATHS[@]}"; do
  [[ -d "$p" ]] || continue
  matches=$(ls -1 "$p" 2>/dev/null | grep -i '^pyvista' || true)
  if [[ -n "$matches" ]]; then
    echo "--- $p ---"
    echo "$matches" | head -10
  fi
done

section "system module: vtk"
errsink module spider vtk 2>&1 | head -40 || true
echo "---"
errsink module avail vtk 2>&1 | head -20 || true

section "system module: anything *vtk*"
errsink module --terse avail 2>&1 | grep -i vtk | head -20 || true

section "system libs: ldconfig hits for libvtk*"
ldconfig -p 2>/dev/null | grep -i libvtk | head -10 || true

section "PyPI reachability (compute node may block this)"
for url in https://pypi.org/simple/vtk/ https://download.pytorch.org/whl/cu118/ https://files.pythonhosted.org/; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$url" || echo "TIMEOUT")
  echo "  $url  ->  $code"
done

section "pip config (env vars + global config)"
env | grep -E '^PIP_' | head -10 || echo "(no PIP_* env vars)"
errsink pip config list 2>&1 | head -10 || true

section "currently installed: torch / torch_geometric / lips / airfrans / pyvista / vtk"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  for pkg in torch torch_geometric lips-benchmark airfrans pyvista vtk dill; do
    ver=$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')
    if [[ -n "$ver" ]]; then echo "  $pkg == $ver"; else echo "  $pkg :: NOT INSTALLED"; fi
  done
else
  echo "(venv not active — run: source env/bin/activate)"
fi

section "import probes (does each module load? exit code only)"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  for mod in torch torch_geometric pyvista vtk airfrans lips.benchmark.airfransBenchmark; do
    if python -c "import $mod" 2>/dev/null; then
      echo "  $mod  ->  OK"
    else
      err=$(python -c "import $mod" 2>&1 | tail -1)
      echo "  $mod  ->  FAIL  (${err:0:120})"
    fi
  done
else
  echo "(venv not active — run: source env/bin/activate)"
fi

section "summary"
WH_VTK_FOUND=0
for p in "${WH_PATHS[@]}"; do
  [[ -d "$p" ]] || continue
  if ls -1 "$p" 2>/dev/null | grep -qi '^vtk-'; then WH_VTK_FOUND=1; fi
done
MOD_VTK_FOUND=0
errsink module spider vtk 2>&1 | grep -qiE '^\s*vtk' && MOD_VTK_FOUND=1
PYPI_OK=0
[[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 https://pypi.org/simple/vtk/ || true)" == "200" ]] && PYPI_OK=1

echo "wheelhouse has standalone vtk wheel: $([[ $WH_VTK_FOUND -eq 1 ]] && echo YES || echo no)"
echo "system 'module load vtk' available:  $([[ $MOD_VTK_FOUND -eq 1 ]] && echo YES || echo no)"
echo "PyPI reachable from this host:       $([[ $PYPI_OK -eq 1 ]] && echo YES || echo no)"
echo
echo "Send the full output of this script back."

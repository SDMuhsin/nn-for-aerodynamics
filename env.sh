# Source this from the project root when starting a new shell.
#
#   source env.sh
#
# It loads the required system modules (on Compute Canada) and activates the
# venv. On CC the modules are mandatory because vtk's Python bindings come from
# the system module (`module load vtk/9.3.0`), not from pip — see
# scripts/setup_cc.sh header for the diagnosed root cause.
#
# Also: HF/Torch caches → ./data/, src/ → PYTHONPATH so `python -m geompnn.train`
# works.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -d /cvmfs/soft.computecanada.ca ]]; then
  # Lmod's bash function is unsafe under `set -u` — silently aborts on the
  # internal `${VAR}` references. Save and disable nounset across the load.
  __nounset_was_on=0
  case $- in *u*) __nounset_was_on=1; set +u ;; esac
  module load StdEnv/2023 gcc/12.3 python/3.11 vtk/9.3.0
  [[ "$__nounset_was_on" -eq 1 ]] && set -u
  unset __nounset_was_on
fi

source "$PROJECT_ROOT/env/bin/activate"

export HF_HOME="$PROJECT_ROOT/data/hf"
export TORCH_HOME="$PROJECT_ROOT/data/torch"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

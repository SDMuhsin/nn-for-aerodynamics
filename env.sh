# Source this from the project root when starting a new shell.
#
#   source env.sh
#
# It activates the venv, exports HF/Torch caches into ./data/ so we keep root
# clean, and adds src/ to PYTHONPATH so `python -m geompnn.train ...` works.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

source "$PROJECT_ROOT/env/bin/activate"

export HF_HOME="$PROJECT_ROOT/data/hf"
export TORCH_HOME="$PROJECT_ROOT/data/torch"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

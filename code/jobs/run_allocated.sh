#!/usr/bin/env bash
# Use only inside an existing confirmed single-GPU allocation.
# No account/partition/qos is guessed, and CUDA_VISIBLE_DEVICES is preserved.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${SLURM_JOB_ID:?Run inside an allocated GPU Slurm job}"
PROJECT_PYTHON="${PROJECT_PYTHON:-$PROJECT_ROOT/.venv-gpu/bin/python}"
"$PROJECT_PYTHON" "$PROJECT_ROOT/code/preflight.py"
"$PROJECT_PYTHON" "$PROJECT_ROOT/code/run.py" "$@"

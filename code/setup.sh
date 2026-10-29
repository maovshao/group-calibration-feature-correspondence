#!/usr/bin/env bash
# Prepare an isolated analysis or Linux/CUDA environment; fitting is a separate command.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PYTHON="${PROJECT_PYTHON:-python3}"
MODE="${1:-}"
if [[ "$MODE" != "--analysis" && "$MODE" != "--gpu" ]] || [[ $# -ne 1 ]]; then
    printf 'Usage: bash code/setup.sh --analysis | --gpu\n' >&2
    exit 2
fi
if [[ "$MODE" == "--gpu" ]]; then
    "$PROJECT_PYTHON" -c 'import sys; assert sys.platform == "linux" and sys.version_info[:2] == (3, 11), "GPU reproduction requires Linux and Python 3.11"'
    VENV_PATH="$PROJECT_ROOT/.venv-gpu"
else
    "$PROJECT_PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12)), "The analysis lock supports Python 3.11 or 3.12"'
    VENV_PATH="$PROJECT_ROOT/.venv-analysis"
fi
if [[ ! -d "$VENV_PATH" ]]; then
    "$PROJECT_PYTHON" -m venv "$VENV_PATH"
fi
VENV_PYTHON="$VENV_PATH/bin/python"
if [[ "$MODE" == "--analysis" ]]; then
    "$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements/analysis.txt"
else
    COMMIT=1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88
    UPSTREAM="$PROJECT_ROOT/code/upstream"
    if [[ ! -d "$UPSTREAM" ]]; then
        mkdir -p "$PROJECT_ROOT/outputs/setup"
        FETCH_DIR=$(mktemp -d "$PROJECT_ROOT/outputs/setup/fetch_XXXXXXXX")
        git clone --no-checkout https://github.com/ThyrixYang/group_calibration.git "$FETCH_DIR/upstream"
        git -C "$FETCH_DIR/upstream" checkout --detach "$COMMIT"
        mv "$FETCH_DIR/upstream" "$UPSTREAM"
        rmdir "$FETCH_DIR"
    fi
    test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$UPSTREAM" status --porcelain --untracked-files=all)"
    "$VENV_PYTHON" - "$PROJECT_ROOT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = json.loads((root / "results/provenance.json").read_text())["data_sha256"]
for split in ("val", "test"):
    path = root / "code/upstream/datasets" / f"cifar10_resnet152_0.9576_{split}.pkl"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected[split]:
        raise SystemExit(f"Cache hash mismatch: {split}")
    print(f"Verified {split} cache: {actual}")
PY
    "$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements/gpu-lock.txt" \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu118
fi
"$VENV_PYTHON" -m pip check
mkdir -p "$PROJECT_ROOT/outputs/setup"
RECEIPT=$(mktemp "$PROJECT_ROOT/outputs/setup/installed_${MODE#--}_XXXXXXXX")
"$VENV_PYTHON" -m pip freeze > "$RECEIPT"
printf '\nEnvironment ready: %s\nInstalled versions: %s\n' "$VENV_PATH" "$RECEIPT"
if [[ "$MODE" == "--gpu" ]]; then
    printf 'Before fitting, enter a Slurm GPU allocation and run: %s code/preflight.py\n' "$VENV_PYTHON"
fi

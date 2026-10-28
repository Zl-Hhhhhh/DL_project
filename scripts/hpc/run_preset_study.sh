#!/usr/bin/env bash
# UniScheduler: NVIDIA A800-SXM4-80GB, image pytorch:py310-2.1.0.
# Run from the project root inside an allocated GPU job.
set -euo pipefail

if [[ ! -f configs/dev.yaml || ! -f data/photo120/manifest.json ]]; then
    printf '%s\n' 'Run from the project root after transferring data/photo120.' >&2
    exit 1
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_AUTOINSTALL=false
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"

PRESET_PYTHON="${PRESET_PYTHON:-python}"
PRESET_RUN_ID="${PRESET_RUN_ID:-hpc_presets_$(date +%Y%m%d_%H%M%S)}"
PRESET_SIZE="${PRESET_SIZE:-512}"

"$PRESET_PYTHON" scripts/hpc/check_environment.py \
    --output ".cache/hpc-environments/$PRESET_RUN_ID.json"
"$PRESET_PYTHON" scripts/prepare_photo_dataset.py
"$PRESET_PYTHON" scripts/run_preset_study.py \
    --manifest data/photo120/manifest.json \
    --device 0 --size "$PRESET_SIZE" \
    --output "experiments/$PRESET_RUN_ID"

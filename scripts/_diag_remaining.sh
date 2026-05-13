#!/bin/bash
# Sequential diag runs for the 13 remaining scene/mode combos.
# chess/calib was already done as smoke test.
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mast3r-slam

DIAG_ROOT="logs/diag"
DATASET_ROOT="datasets/7-scenes"
LOG_ROOT="logs/diag_runs"
mkdir -p "$LOG_ROOT"

run_one() {
    local scene="$1"
    local mode="$2"
    local cfg="config/eval_${mode}.yaml"
    local diag_dir="$DIAG_ROOT/$scene/$mode"
    local log_file="$LOG_ROOT/${scene}_${mode}.log"
    mkdir -p "$diag_dir"
    echo "[$(date +%H:%M:%S)] START $scene/$mode"
    python main.py \
        --dataset "$DATASET_ROOT/$scene" \
        --no-viz \
        --config "$cfg" \
        --save-as "diag/$scene/$mode" \
        --diag-dir "$diag_dir" \
        --diag-stride 2 \
        > "$log_file" 2>&1
    # Quick sanity check
    python -c "
import sys, numpy as np
from pathlib import Path
p = Path('$diag_dir/tracking.npz')
if not p.exists():
    print('FAIL: tracking.npz missing'); sys.exit(1)
d = np.load(p, allow_pickle=True)
n_pair = int(d['n_pixels'].size)
finite = int(np.isfinite(d['err']).sum())
print(f'[$(date +%H:%M:%S)] DONE  $scene/$mode  pairs={n_pair}  finite={finite:,}')
if finite < 1000:
    sys.exit(1)
"
}

run_one chess no_calib
for scene in fire heads office pumpkin redkitchen stairs; do
    for mode in calib no_calib; do
        run_one "$scene" "$mode"
    done
done

echo "[$(date +%H:%M:%S)] ALL_DONE"

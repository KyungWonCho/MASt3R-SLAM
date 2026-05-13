#!/bin/bash
# Run 7-Scenes diag collection sequentially: chess first (smoke check), then
# remaining scenes in calib + no-calib. After all runs, runs analyze_diag.py.
#
# Usage: bash scripts/run_diag_chess.sh [stride]
#
# Outputs:
#   logs/diag/<scene>/<mode>/tracking.npz
#   logs/diag/<scene>/<mode>/loop.npz   (when loop edges were added)
#   logs/diag/plots/                    (PNG + summary.csv + summary.md)
#   logs/7-scenes/diag/<scene>/<mode>/  (SLAM outputs: traj + ply + stats)

set -euo pipefail

STRIDE="${1:-2}"
DIAG_ROOT="logs/diag"
DATASET_ROOT="datasets/7-scenes"
LOG_ROOT="logs/diag_runs"

mkdir -p "$LOG_ROOT" "$DIAG_ROOT"

run_one() {
    local scene="$1"
    local mode="$2"  # calib | no_calib
    local cfg="config/eval_${mode}.yaml"
    local diag_dir="$DIAG_ROOT/$scene/$mode"
    local save_as="diag/$scene/$mode"
    local log_file="$LOG_ROOT/${scene}_${mode}.log"
    echo
    echo "=========================================================="
    echo "Running  $scene  ($mode)  -> $diag_dir"
    echo "Log: $log_file"
    echo "=========================================================="
    mkdir -p "$diag_dir"
    python main.py \
        --dataset "$DATASET_ROOT/$scene" \
        --no-viz \
        --config "$cfg" \
        --save-as "$save_as" \
        --diag-dir "$diag_dir" \
        --diag-stride "$STRIDE" \
        2>&1 | tee "$log_file"
    # Verify tracking NPZ is reasonable
    python -c "
import sys, numpy as np
from pathlib import Path
p = Path('$diag_dir/tracking.npz')
if not p.exists():
    print('FAIL: tracking.npz missing'); sys.exit(1)
d = np.load(p, allow_pickle=True)
n_pair = int(d['n_pixels'].size)
n_pix = int(d['err'].size)
finite = int(np.isfinite(d['err']).sum())
print(f'  tracking.npz: pairs={n_pair}, pixels={n_pix:,}, finite_err={finite:,}')
if finite < 1000:
    print('FAIL: too few finite err pixels'); sys.exit(1)
"
}

# 1) chess first — smoke
run_one chess calib
run_one chess no_calib

# 2) remaining scenes
for scene in fire heads office pumpkin redkitchen stairs; do
    for mode in calib no_calib; do
        run_one "$scene" "$mode"
    done
done

# 3) analysis
echo
echo "=========================================================="
echo "Running analyze_diag.py"
echo "=========================================================="
python scripts/analyze_diag.py --diag-root "$DIAG_ROOT" --out-dir "$DIAG_ROOT/plots" \
    2>&1 | tee "$LOG_ROOT/analyze.log"

echo
echo "Done. Results:"
echo "  $DIAG_ROOT/plots/summary.md   (overview table)"
echo "  $DIAG_ROOT/plots/summary.csv  (per-row stats)"
echo "  $DIAG_ROOT/plots/*.png         (calibration + loop edge plots)"
echo "  $LOG_ROOT/*.log                (per-run console logs)"

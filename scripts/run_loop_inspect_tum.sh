#!/bin/bash
# Run loop-closure inspector on all locally available TUM sequences.
# Lightweight (vanilla SLAM + per-edge summary); safe on local single GPU.

set -u
export CUDA_VISIBLE_DEVICES=0

ROOT="datasets/tum"
OUT_ROOT="logs/loop_inspect"

mkdir -p "$OUT_ROOT"

for seq_dir in "$ROOT"/rgbd_dataset_*; do
    [ -d "$seq_dir" ] || continue
    seq=$(basename "$seq_dir")
    out_dir="$OUT_ROOT/$seq"
    if [ -f "$out_dir/loop_inspect.npz" ]; then
        echo "SKIP $seq (already done)"
        continue
    fi
    mkdir -p "$out_dir"
    echo "=== $seq ==="
    python main.py \
        --dataset "$seq_dir" \
        --no-viz \
        --save-as "inspect/tum/$seq" \
        --config config/eval_no_calib.yaml \
        --loop-inspect-dir "$out_dir" \
        2>&1 | tee "$out_dir/run.log"
done

echo
echo "=========== analysis ==========="
for seq_dir in "$OUT_ROOT"/rgbd_dataset_*; do
    [ -f "$seq_dir/loop_inspect.npz" ] || continue
    seq=$(basename "$seq_dir")
    gt="$ROOT/$seq/groundtruth.txt"
    echo "--- $seq ---"
    if [ -f "$gt" ]; then
        python scripts/analyze_loop_inspect.py \
            --inspect-dir "$seq_dir" --gt "$gt" 2>&1 | tail -20
    else
        python scripts/analyze_loop_inspect.py \
            --inspect-dir "$seq_dir" 2>&1 | tail -20
    fi
done

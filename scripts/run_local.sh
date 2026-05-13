#!/bin/bash
# Local single-GPU eval runner. Mirrors the per-gpu server scripts but
# runs serially on the local box. Safe to run *without* --diag-dir
# (the diag D2H pressure is what previously broke the local PCIe link;
# vanilla SLAM eval was fine on this box before).
#
# Pick whichever (scene, variant) combinations you want to add to the
# server's work pool — skip-if-exists ensures no double-work if the
# trajectory file already exists.

set -u
export CUDA_VISIBLE_DEVICES=0

# Edit these lists to taste:
SCENES="chess fire heads"
VARIANTS="vanilla calibonly caponly calibcap fusion"

mkdir -p logs/diag_runs
for variant in $VARIANTS; do
    if [ "$variant" = "vanilla" ]; then
        cfg="config/eval_calib.yaml"
    else
        cfg="config/eval_calib_${variant}.yaml"
    fi
    for scene in $SCENES; do
        save_path="7-scenes/$variant/calib/$scene"
        if [ -f "logs/$save_path/$scene.txt" ]; then
            echo "SKIP $scene/$variant (already done)"
            continue
        fi
        echo "=== $scene / $variant (LOCAL GPU $CUDA_VISIBLE_DEVICES) ==="
        python main.py \
            --dataset "datasets/7-scenes/$scene" \
            --no-viz \
            --save-as "$save_path" \
            --config "$cfg" \
            2>&1 | tee "logs/diag_runs/eval_${variant}_${scene}.log"
    done
done
echo "Local done."

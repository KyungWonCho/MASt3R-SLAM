#!/bin/bash
# 7-Scenes vanilla + ablation eval, GPU 3 share.
# Scenes: stairs only — lightest GPU; pick up extras from other GPUs once done.

set -u
export CUDA_VISIBLE_DEVICES=3
SCENES="stairs"
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
        echo "=== $scene / $variant (GPU $CUDA_VISIBLE_DEVICES) ==="
        python main.py \
            --dataset "datasets/7-scenes/$scene" \
            --no-viz \
            --save-as "$save_path" \
            --config "$cfg" \
            2>&1 | tee "logs/diag_runs/eval_${variant}_${scene}.log"
    done
done
echo "GPU 3 done."

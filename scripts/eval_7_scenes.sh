#!/bin/bash
dataset_path="datasets/7-scenes/"
datasets=(
    chess
    fire
    heads
    office
    pumpkin
    redkitchen
    stairs
)

variant=""
no_calib=false
print_only=false
robust=false
edge_fusion=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        --robust)
            robust=true
            ;;
        --edge-fusion)
            edge_fusion=true
            ;;
        --variant)
            variant="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$no_calib" = true ]; then
    split="no_calib"
    cfg="config/eval_no_calib.yaml"
else
    split="calib"
    cfg="config/eval_calib.yaml"
fi
if [ "$robust" = true ]; then
    cfg="${cfg%.yaml}_robust.yaml"
elif [ "$edge_fusion" = true ]; then
    cfg="${cfg%.yaml}_edgefusion.yaml"
elif [ -n "$variant" ] && [ "$variant" != "vanilla" ]; then
    cand="${cfg%.yaml}_${variant}.yaml"
    if [ -f "$cand" ]; then
        cfg="$cand"
    fi
fi
if [ -z "$variant" ]; then
    if [ "$robust" = true ]; then
        variant="robust"
    elif [ "$edge_fusion" = true ]; then
        variant="edgefusion"
    else
        variant="vanilla"
    fi
fi
save_path="7-scenes/$variant/$split"
logs_dir="logs/$save_path"

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        python main.py --dataset "$dataset_name" --no-viz \
            --save-as "$save_path/$dataset" --config "$cfg"
    done
fi

echo
echo "=========== 7-Scenes [$variant/$split] — geometry + ATE ==========="
for dataset in ${datasets[@]}; do
    echo
    echo "--- $dataset ---"
    python scripts/eval_geometry.py \
        --est-ply  "$logs_dir/$dataset/$dataset.ply" \
        --est-traj "$logs_dir/$dataset/$dataset.txt" \
        --scene-dir "$dataset_path$dataset"
done

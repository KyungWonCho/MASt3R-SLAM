#!/bin/bash
dataset_path="datasets/tum/"
datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
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
save_path="tum/$variant/$split"
logs_dir="logs/$save_path"

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        python main.py --dataset "$dataset_name" --no-viz \
            --save-as "$save_path/$dataset" --config "$cfg"
    done
fi

echo
echo "=========== TUM [$variant/$split] — ATE ==========="
for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo
    echo "--- $dataset ---"
    evo_ape tum "$dataset_name/groundtruth.txt" "$logs_dir/$dataset/$dataset.txt" -as
done

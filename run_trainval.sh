#!/bin/bash

#SBATCH --job-name=rcf_instrument_trainval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_rcf_instrument_trainval_%j.out


source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "4 runs training on ALL 8 datasets with different validation splits"
echo "Started at: $(date)"
echo "=========================================="

VAL_SPLITS=(
    "fold1_val.txt"
    "fold2_val.txt"
    "fold3_val.txt"
    "fold4_val.txt"
)

for i in 0 1 2 3; do
    VAL=${VAL_SPLITS[$i]}
    RUN_DIR="saved/saved_instrument_trainval_run${i}"

    echo ""
    echo "=========================================="
    echo "Run $((i+1)) / 4"
    echo "Validation: ${VAL}"
    echo "Checkpoint dir: ${RUN_DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 python main.py \
        configs/instrument/rcf_instrument_trainval.yaml \
        --opts \
            test_dataset_kwargs.split ${VAL} \
            checkpoints_dir ${RUN_DIR} \
            allow_overwriting_checkpoints_dir True

    if [ $? -ne 0 ]; then
        echo "Run $((i+1)) failed!"
        exit 1
    fi

    echo "Run $((i+1)) finished at: $(date)"
done

echo ""
echo "=========================================="
echo "ALL 4 TRAINING RUNS FINISHED"
echo "=========================================="
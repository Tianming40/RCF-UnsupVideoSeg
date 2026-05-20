#!/bin/bash

#SBATCH --job-name=rcf_instrument_trainval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm_rcf_instrument_trainval_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "Training on all 8 datasets (trainval)"
echo "Started at: $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python main.py configs/instrument/rcf_instrument_trainval.yaml

if [ $? -eq 0 ]; then
    echo "Training completed at: $(date)"
else
    echo "Training failed!"
    exit 1
fi

# echo ""
# echo "=========================================="
# echo "Testing on all 4 folds"
# echo "=========================================="

# #  best checkpoint
# CKPT=$(ls saved/saved_instrument_trainval/*.ckpt | grep -v last | head -1)
# echo "Using checkpoint: $CKPT"

# for fold in 1 2 3 4; do
#     echo ""
#     echo "--- Testing fold $fold ---"
#     CUDA_VISIBLE_DEVICES=0 python main.py \
#         configs/instrument/rcf_instrument_trainval.yaml \
#         --test \
#         --test-override-pretrained "$CKPT" \
#         --opts allow_overwriting_checkpoints_dir True \
#                eval_pos_th -1 \
#                test_dataset_kwargs.split fold${fold}_val.txt
# done

# echo ""
# echo "=========================================="
# echo "All done! Finished at: $(date)"
# echo "=========================================="

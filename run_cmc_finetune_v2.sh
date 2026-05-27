#!/bin/bash

#SBATCH --job-name=rcf_cmc_finetune_v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_rcf_cmc_finetune_v2_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "Fine-tuning V2 on CMC_grasp10 (topk=4, boundary_threshold=π/18)"
echo "Resumed from data_medical checkpoint"
echo "Started at: $(date)"
echo "=========================================="

TIMESTAMP=$(date +%m%d_%H%M%S)
RUN_DIR="saved/saved_cmc_finetune_v2_${TIMESTAMP}"

echo "Checkpoint dir: ${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    configs/instrument/rcf_cmc_finetune_v2.yaml \
    --resume saved_this_use8datatrain_3best_test_MITI/saved_instrument_trainval_run0/epoch=7-step=1800.ckpt \
    --opts checkpoints_dir "${RUN_DIR}" \
           allow_overwriting_checkpoints_dir True

echo "=========================================="
echo "FINISHED at: $(date)"
echo "=========================================="

#!/bin/bash

#SBATCH --job-name=rcf_cmc_all_v2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_rcf_cmc_all_v2b_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "V2b: mask_size=128, clamp_flow_t=10, free_residual_with_affine=true, epochs=60"
echo "Data: CMC grasp-0+5+10 merged (1443 train pairs)"
echo "Started at: $(date)"
echo "=========================================="

TIMESTAMP=$(date +%m%d_%H%M%S)
RUN_DIR="saved/saved_cmc_all_finetune_v2b_${TIMESTAMP}"

echo "Checkpoint dir: ${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    configs/instrument/rcf_cmc_all_finetune_v2b.yaml \
    --opts checkpoints_dir "${RUN_DIR}" \
           pretrained_model "saved_this_use8datatrain_3best_test_MITI/saved_instrument_trainval_run0/epoch=7-step=1800.ckpt" \
           allow_overwriting_checkpoints_dir True

echo "=========================================="
echo "FINISHED at: $(date)"
echo "=========================================="

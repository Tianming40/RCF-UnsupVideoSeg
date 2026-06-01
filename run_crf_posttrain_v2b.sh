#!/bin/bash

#SBATCH --job-name=rcf_crf_posttrain_v2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_rcf_crf_posttrain_v2b_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "CRF post-training on v2b checkpoint"
echo "Stage 1: MAA channel auto-detection"
echo "Stage 2: short post-training with CRF pseudo-label loss"
echo "Started at: $(date)"
echo "=========================================="

TIMESTAMP=$(date +%m%d_%H%M%S)
RUN_DIR="saved/saved_cmc_crf_posttrain_v2b_${TIMESTAMP}"

CKPT="saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498.ckpt"

echo "Base checkpoint : ${CKPT}"
echo "Output dir      : ${RUN_DIR}"
echo ""

CUDA_VISIBLE_DEVICES=0 python main_crf_posttrain.py \
    configs/instrument/rcf_cmc_crf_posttrain_v2b.yaml \
    --ckpt        "${CKPT}"    \
    --output_dir  "${RUN_DIR}" \
    --crf_epochs  10           \
    --crf_lr      2e-5         \
    --w_crf       0.5          \
    --crf_pos_weight 2.0       \
    --crf_neg_weight 1.0       \
    --crf_iters   10           \
    --crf_sxy     60.          \
    --crf_srgb    5.           \
    --probe_frames 20          \
    --probe_th    0.35

echo "=========================================="
echo "FINISHED at: $(date)"
echo "=========================================="

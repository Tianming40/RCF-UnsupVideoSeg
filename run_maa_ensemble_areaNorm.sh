#!/bin/bash
# MAA + Flow ensemble inference with area-normalised flow scoring.
# Checkpoint: saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498 (65.21% best)
# Architecture: V2b (mask_size=128, clamp_flow_t=10, free_residual_with_affine)
#
# Key difference from run_maa_ensemble_v2b.sh:
#   Flow score is penalised by log(mask_area) to remove the size bias that
#   causes the scorer to trivially prefer smaller (higher-threshold) masks.
#   flow_norm = flow_raw + BETA * log(frac)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_maa_ensemble_areaNorm.sh
#
# Optional env-var overrides:
#   BETA=2.0   bash run_maa_ensemble_areaNorm.sh   # stronger area penalty
#   BETA=0.0   bash run_maa_ensemble_areaNorm.sh   # disable (reproduces original)
#   ALPHA=0.7  bash run_maa_ensemble_areaNorm.sh   # more weight on MAA
#   ARGMAX=1   bash run_maa_ensemble_areaNorm.sh   # argmax instead of softmax blend

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

export LD_LIBRARY_PATH=/home/tianming/anaconda3/envs/rcf/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

CKPT="saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498.ckpt"
CONFIG="configs/instrument/maa_ensemble_v2b_areaNorm.yaml"
TIMESTAMP=$(date +%m%d_%H%M%S)
OUTPUT="saved/maa_areaNorm_v2b_${TIMESTAMP}"

ALPHA=${ALPHA:-0.7}   # 0.7 = 70% MAA + 30% flow; flow质量差时推荐偏向MAA
BETA=${BETA:-1.0}

echo "=========================================="
echo "MAA + Flow Ensemble (z-norm + argmax)"
echo "Checkpoint  : $CKPT"
echo "Output      : $OUTPUT"
echo "Alpha(MAA)  : $ALPHA   Flow: $(echo "1 - $ALPHA" | bc)"
echo "Area beta   : $BETA    (0 = no area normalisation)"
echo "Started at  : $(date)"
echo "=========================================="

# Optional flags (uncomment to enable):
# TTA_FLAG="--tta"
# CRF_FLAG="--use_crf"
# EXTRA="--extra_ckpts saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=30-step=11191.ckpt"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python tools/maa_inference_ensemble_areaNorm.py \
    --config          "$CONFIG" \
    --ckpt            "$CKPT" \
    --output          "$OUTPUT" \
    --use_flow \
    --alpha           "$ALPHA" \
    --flow_area_beta  "$BETA" \
    --thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 \
    --dino_arch       vit_small \
    --dino_patch_size 8 \
    --dino_tau        0.2 \
    --workers         8 \
    ${TTA_FLAG:-} \
    ${CRF_FLAG:-} \
    ${EXTRA:-}

echo "=========================================="
echo "FINISHED at: $(date)"
echo "Results saved to: $OUTPUT"
echo "=========================================="

#!/bin/bash
# MAA + Flow weighted inference ensemble
# Checkpoint: saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498 (65.21% best)
# Architecture: V2b (mask_size=128, clamp_flow_t=10, free_residual_with_affine)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_maa_ensemble_v2b.sh
#
# Optional overrides (pass as env vars):
#   ALPHA=0.7   bash run_maa_ensemble_v2b.sh   # more weight on MAA
#   ALPHA=0.3   bash run_maa_ensemble_v2b.sh   # more weight on flow

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

CKPT="saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498.ckpt"
CONFIG="configs/instrument/maa_ensemble_v2b.yaml"
TIMESTAMP=$(date +%m%d_%H%M%S)
OUTPUT="saved/maa_ensemble_v2b_${TIMESTAMP}"

# MAA-flow balance: 0.5 = equal weight, 1.0 = MAA only, 0.0 = flow only
ALPHA=${ALPHA:-0.5}

echo "=========================================="
echo "MAA + Flow Ensemble Inference"
echo "Checkpoint : $CKPT"
echo "Output     : $OUTPUT"
echo "Alpha(MAA) : $ALPHA   Flow: $(echo "1 - $ALPHA" | bc)"
echo "Started at : $(date)"
echo "=========================================="

# Optional flags (uncomment to enable):
# TTA_FLAG="--tta"
# CRF_FLAG="--use_crf"
# EXTRA="--extra_ckpts saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=30-step=11191.ckpt"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python tools/maa_inference_ensemble.py \
    --config    "$CONFIG" \
    --ckpt      "$CKPT" \
    --output    "$OUTPUT" \
    --use_flow \
    --alpha     "$ALPHA" \
    --thresholds 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 \
    --maa_temperature 10.0 \
    --eval_pos_th 0.35 \
    --dino_arch vit_small \
    --dino_patch_size 8 \
    --dino_tau 0.2 \
    --workers 8 \
    ${TTA_FLAG:-} \
    ${CRF_FLAG:-} \
    ${EXTRA:-}

echo "=========================================="
echo "FINISHED at: $(date)"
echo "Results saved to: $OUTPUT"
echo "=========================================="

#!/bin/bash
# Sweep eval_pos_th on the best checkpoint to find optimal threshold.
# Run this while training is happening on another GPU/process.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash sweep_pos_th.sh
#   CUDA_VISIBLE_DEVICES=0 bash sweep_pos_th.sh   (if GPU0 has spare capacity)

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

CKPT="saved/saved_cmc_merged_finetune_v2_0528_093615/epoch=9-step=2282.ckpt"
CONFIG="configs/instrument/rcf_cmc_finetune_v2.yaml"
SWEEP_DIR="saved/sweep_pos_th_$(date +%m%d_%H%M%S)"

echo "Checkpoint: $CKPT"
echo "Sweep dir:  $SWEEP_DIR"
echo ""
echo "th    | val_miou"
echo "------+----------"

for TH in 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55; do
    OUT_DIR="${SWEEP_DIR}/th_${TH}"

    LOG=$(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_v2.py "$CONFIG" \
        --test \
        --opts allow_overwriting_checkpoints_dir True \
               checkpoints_dir "$OUT_DIR" \
               pretrained_model "$CKPT" \
               disable_wandb true \
               batch_size 1 \
               eval_pos_th "$TH" \
        2>&1)

    MIOU=$(echo "$LOG" | grep "val_miou:" | grep -v frame_avg | tail -1 | awk '{print $NF}')
    echo "$TH   | ${MIOU:-ERROR}"
done

echo ""
echo "Done. Full logs in $SWEEP_DIR"

#!/bin/bash
# Sweep eval_pos_th on CMC val set using the best checkpoint.
# Saves segmentation images for each threshold so you can visually compare.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash sweep_pos_th_cmc.sh

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

CKPT="saved/saved_cmc_merged_finetune_v2_0528_093615/epoch=9-step=2282.ckpt"
TIMESTAMP=$(date +%m%d_%H%M%S)

echo "Checkpoint: $CKPT"
echo "Results:    saved/sweep_pos_th_cmc_${TIMESTAMP}/th_XX/saved_eval/"
echo ""

for TH in 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55; do
    OUT_DIR="saved/sweep_pos_th_cmc_${TIMESTAMP}/th_${TH}"

    echo "=============================="
    echo "eval_pos_th = $TH  →  $OUT_DIR"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_v2.py \
        configs/instrument/test_cmc_val_v2.yaml \
        --test \
        --opts allow_overwriting_checkpoints_dir True \
               checkpoints_dir "$OUT_DIR" \
               pretrained_model "$CKPT" \
               disable_wandb true \
               batch_size 1 \
               eval_pos_th "$TH"

    if [ $? -ne 0 ]; then
        echo "Failed at th=$TH"
        exit 1
    fi
done

echo ""
echo "=============================="
echo "Done! Compare images in:"
echo "  saved/sweep_pos_th_cmc_${TIMESTAMP}/th_*/saved_eval/"
echo "=============================="

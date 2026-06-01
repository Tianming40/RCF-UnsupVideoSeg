#!/bin/bash

#SBATCH --job-name=rcf_eval_crf_posttrain_v2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_rcf_eval_crf_posttrain_v2b_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

# CRF post-training checkpoints from saved_cmc_crf_posttrain_v2b_0601_110702
# Ranked by val_miou on data_medical/fold1_val:
#   epoch=1:  62.93%  ← best
#   epoch=3:  62.14%  ← 2nd
#   last:     61.54%  (epoch=9)
CKPTS=(
    "saved/saved_cmc_crf_posttrain_v2b_0601_110702/epoch=1-step=722.ckpt"
    "saved/saved_cmc_crf_posttrain_v2b_0601_110702/epoch=3-step=1444.ckpt"
    "saved/saved_cmc_crf_posttrain_v2b_0601_110702/last.ckpt"
)

TIMESTAMP=$(date +%m%d_%H%M%S)

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .ckpt)
    CKPT_EVAL_DIR="saved/eval_crf_posttrain_v2b/${TIMESTAMP}/${CKPT_NAME}"

    echo ""
    echo "=============================="
    echo "Evaluating: $CKPT"
    echo "Output dir: $CKPT_EVAL_DIR"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=0 python main_v2.py \
        configs/instrument/test_cmc_val_v2b.yaml \
        --test \
        --opts allow_overwriting_checkpoints_dir True \
               checkpoints_dir "$CKPT_EVAL_DIR" \
               pretrained_model "$CKPT" \
               disable_wandb true \
               batch_size 1

    if [ $? -ne 0 ]; then
        echo "Evaluation of $CKPT failed!"
        exit 1
    fi
done

echo ""
echo "=============================="
echo "All done! Results saved to saved/eval_crf_posttrain_v2b/${TIMESTAMP}/"
echo "=============================="

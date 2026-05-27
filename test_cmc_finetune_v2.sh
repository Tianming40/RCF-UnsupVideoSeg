#!/bin/bash

#SBATCH --job-name=rcf_eval_cmc_v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_rcf_eval_cmc_v2_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

# V2 checkpoints from saved_cmc_finetune_v2_0527_150009
# (correct deinterlace + normalized flows _NewCT_nm)
# Ranked by val_miou on data_medical/fold1_val:
#   epoch=14: 64.67%  ← best
#   epoch=8:  62.80%  ← 2nd
#   last:     28.65%  (epoch 28)
CKPTS=(
    "saved/saved_cmc_finetune_v2_0527_150009/epoch=14-step=2647.ckpt"
    "saved/saved_cmc_finetune_v2_0527_150009/epoch=8-step=1921.ckpt"
    "saved/saved_cmc_finetune_v2_0527_150009/last.ckpt"
)

TIMESTAMP=$(date +%m%d_%H%M%S)

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .ckpt)
    CKPT_EVAL_DIR="saved/eval_cmc_v2/${TIMESTAMP}/${CKPT_NAME}"

    echo ""
    echo "=============================="
    echo "Evaluating: $CKPT"
    echo "Output dir: $CKPT_EVAL_DIR"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=0 python main_v2.py \
        configs/instrument/test_cmc_val_v2.yaml \
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
echo "All done! Results saved to saved/eval_cmc_v2/${TIMESTAMP}/"
echo "=============================="

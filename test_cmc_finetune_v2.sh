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

# V2 checkpoints from saved_cmc_all_finetune_v2_0528_103539
# (topk=4, grasp-0+5+10 merged, correct deinterlace, original flows _NewCT)
# Ranked by val_miou on data_medical/fold1_val:
#   epoch=8:  68.95%  ← best
#   epoch=15: 68.46%  ← 2nd
#   last:     62.12%  (epoch 29)
CKPTS=(
    "saved/saved_cmc_all_finetune_v2_0528_103539/epoch=8-step=2161.ckpt"
    "saved/saved_cmc_all_finetune_v2_0528_103539/epoch=15-step=4688.ckpt"
    "saved/saved_cmc_all_finetune_v2_0528_103539/last.ckpt"
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

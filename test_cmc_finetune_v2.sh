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

# V2b checkpoints from saved_cmc_all_finetune_v2b_0528_132020
# (mask_size=128, clamp_flow_t=10, free_residual_with_affine=True, grasp-0+5+10)
# Ranked by val_miou on data_medical/fold1_val:
#   epoch=17: 65.21%  ← best
#   epoch=30: 64.14%  ← 2nd
#   last:     57.38%  (epoch 59)
CKPTS=(
    "saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498.ckpt"
    "saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=30-step=11191.ckpt"
    "saved/saved_cmc_all_finetune_v2b_0528_132020/last.ckpt"
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
echo "All done! Results saved to saved/eval_cmc_v2/${TIMESTAMP}/"
echo "=============================="

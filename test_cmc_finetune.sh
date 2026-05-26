#!/bin/bash

#SBATCH --job-name=rcf_eval_cmc_finetune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_rcf_eval_cmc_finetune_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

# CMC fine-tuned checkpoints (deinterlaced, ranked by val_miou on data_medical)
# epoch=26: 69.08%, epoch=27: 68.53%, epoch=13: 67.30%
# + original data_medical checkpoints (for comparison)
CKPTS=(
    "saved/saved_cmc_finetune_0526_132131/epoch=26-step=4099.ckpt"
    "saved/saved_cmc_finetune_0526_132131/epoch=27-step=4220.ckpt"
    "saved/saved_cmc_finetune_0526_132131/epoch=13-step=2526.ckpt"
    "saved_this_use8datatrain_3best_test_MITI/saved_instrument_trainval_run0/epoch=7-step=1800.ckpt"
    "saved_this_use8datatrain_3best_test_MITI/saved_instrument_trainval_run0/epoch=9-step=2250.ckpt"
    "saved_this_use8datatrain_3best_test_MITI/saved_instrument_trainval_run3/epoch=35-step=8100.ckpt"
)

TIMESTAMP=$(date +%m%d_%H%M%S)

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .ckpt)
    CKPT_EVAL_DIR="saved/eval_cmc_finetune/${TIMESTAMP}/${CKPT_NAME}"

    echo ""
    echo "=============================="
    echo "Evaluating: $CKPT"
    echo "Output dir: $CKPT_EVAL_DIR"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=0 python main.py \
        configs/instrument/test_cmc_val.yaml \
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
echo "All done! Results saved to saved/eval_cmc_finetune/${TIMESTAMP}/"
echo "=============================="

#!/bin/bash

#SBATCH --job-name=rcf_eval_best3_cmc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_rcf_eval_best3_cmc_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg


CKPTS=(
    "saved/saved_instrument_trainval_run0/epoch=7-step=1800.ckpt"
    "saved/saved_instrument_trainval_run0/epoch=9-step=2250.ckpt"
    "saved/saved_instrument_trainval_run3/epoch=35-step=8100.ckpt"
)

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .ckpt)
    CKPT_EVAL_DIR="saved/eval_best3_on_cmc/${CKPT_NAME}"
    
    echo ""
    echo "=============================="
    echo "Evaluating: $CKPT"
    echo "Output dir: $CKPT_EVAL_DIR"
    echo "=============================="
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        configs/instrument/test_MITI.yaml \
        --test \
        --opts allow_overwriting_checkpoints_dir True \
               checkpoints_dir "$CKPT_EVAL_DIR" \
               disable_wandb true
done

echo ""
echo "=============================="
echo "All done! Results saved to saved/eval_best3_on_cmc/"
echo "=============================="
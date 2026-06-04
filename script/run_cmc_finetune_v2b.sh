#!/bin/bash
#SBATCH --job-name=cmc_finetune_v2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/script/slurm_out/slurm_cmc_finetune_v2b_%j.out

# Finetune on CMC grasp-10, starting from phase2 epoch=32 checkpoint.
# V2b: mask_size=128, clamp_flow_t=10, free_residual_with_affine=True.
#
# Usage: sbatch script/run_cmc_finetune_v2b.sh
#        bash   script/run_cmc_finetune_v2b.sh [gpu]

GPU=${1:-0}

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

TIMESTAMP=$(date +"%y%m%d_%H%M%S")
RUN_DIR="saved/cmc_grasp10_finetune_v2b_${TIMESTAMP}"

echo "=========================================="
echo "CMC grasp-10 finetune (V2b)"
echo "Pretrained: saved/phase2_260604_081848/epoch=32-step=7425.ckpt"
echo "Config    : configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml"
echo "Run dir   : ${RUN_DIR}"
echo "Started   : $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU} python main_v2.py \
    configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml \
    --opts \
        checkpoints_dir "${RUN_DIR}" \
        allow_overwriting_checkpoints_dir True

echo "Done: $(date)"

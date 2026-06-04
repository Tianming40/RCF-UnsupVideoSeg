#!/bin/bash
#SBATCH --job-name=cmc_vis_last
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/script/slurm_out/slurm_cmc_vis_last_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

TIMESTAMP=$(date +"%y%m%d_%H%M%S")
OUTPUT="saved/cmc_vis_last_${TIMESTAMP}"

echo "Output: ${OUTPUT}/saved_eval"
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=0 python tools/cmc_vis_inference.py \
    --config        configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml \
    --ckpt          saved/cmc_grasp10_finetune_v2b_260604_120527/last.ckpt \
    --output        "${OUTPUT}" \
    --split         ImageSets/val.txt \
    --union_channels 0 2 3 \
    --workers       0

echo "Done: $(date)"
echo "Saved to: ${OUTPUT}/saved_eval"

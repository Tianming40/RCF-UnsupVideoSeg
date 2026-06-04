#!/bin/bash

#SBATCH --job-name=rcf_kfold_phase2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/script/slurm_out/slurm_kfold_phase2_%j.out

# Usage: bash script/run_phase2_final.sh <BEST_EPOCH> [RUN_TAG]
# BEST_EPOCH : average best epoch from Phase 1 (output of run_phase1_kfold.sh)
# RUN_TAG    : optional tag from Phase 1 run (YYMMDD_HHMMSS); auto-generated if omitted
# Examples:
#   bash script/run_phase2_final.sh 65 260603_120000
#   bash script/run_phase2_final.sh 65   # generates its own tag

BEST_EPOCH=${1:-45}
RUN_TAG=${2:-$(date +"%y%m%d_%H%M%S")}

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

echo "=========================================="
echo "Phase 2: Final model on ALL 8 datasets (mask_layer=5)"
echo "Train: trainval.txt (all 8 datasets, no held-out)"
echo "Epoch: ${BEST_EPOCH}  (fixed from Phase 1 k-fold)"
echo "Run tag: ${RUN_TAG}"
echo ""
echo "Checkpoint strategy:"
echo "  - Validation runs every 5 epochs on fold1_val.txt (monitoring only)"
echo "  - Final model = last.ckpt  (NOT best.ckpt, which is biased)"
echo "Started at: $(date)"
echo "=========================================="

RUN_DIR="saved/phase2_${RUN_TAG}"

CUDA_VISIBLE_DEVICES=0 python main.py \
    configs/instrument/rcf_kfold_phase2.yaml \
    --opts \
        epochs ${BEST_EPOCH} \
        checkpoints_dir ${RUN_DIR} \
        allow_overwriting_checkpoints_dir True

if [ $? -ne 0 ]; then
    echo "Phase 2 training failed!"
    exit 1
fi

FINAL_CKPT="${RUN_DIR}/last.ckpt"
echo ""
echo "=========================================="
echo "Phase 2 FINISHED at: $(date)"
echo "Run tag            : ${RUN_TAG}"
echo "Checkpoint dir     : ${RUN_DIR}"
echo "Final model        : ${FINAL_CKPT}"
echo ""
echo "Use this checkpoint for inference:"
echo "  python main.py configs/instrument/rcf_kfold_phase2.yaml \\"
echo "      --test --test-override-pretrained ${FINAL_CKPT}"
echo "=========================================="

#!/bin/bash
# Run inference on CMC dataset for all finetune checkpoints.
# Uses main_v2.py (V2b architecture) with CMC test settings.
#
# Usage: bash script/run_cmc_inference.sh [gpu]

GPU=${1:-0}
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
FINETUNE_DIR="saved/cmc_grasp10_finetune_v2b_260604_120527"
SWEEP_NAME="cmc_g10_infer_${TIMESTAMP}"
CONFIG=configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd "$(dirname "$0")/.."

CKPTS=$(find "${FINETUNE_DIR}" -name "*.ckpt" | sort -V)

echo "=========================================="
echo "CMC inference sweep (V2b finetune ckpts)"
echo "Started: $(date)"
echo "=========================================="

SWEEP_DIR="saved/${SWEEP_NAME}"
mkdir -p "${SWEEP_DIR}"

for CKPT in ${CKPTS}; do
    NAME=$(basename "${CKPT%.ckpt}")
    OUT_DIR="${SWEEP_DIR}/${NAME}"

    echo ""
    echo "--- ${NAME} ---"
    echo "Output: ${OUT_DIR}/saved_eval"

    CUDA_VISIBLE_DEVICES=${GPU} python main_v2.py "${CONFIG}" \
        --test \
        --opts \
            pretrained_model "${CKPT}" \
            checkpoints_dir "${OUT_DIR}" \
            allow_overwriting_checkpoints_dir True \
            data_path /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced \
            test_data_path /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced \
            test_dataset_kwargs.split ImageSets/val.txt \
            test_dataset_kwargs.zero_ann True \
            eval_save True \
            eval_export False \
            export_all_seg False \
            batch_size 1 \
            workers 4

    echo "Done: $(date)"
done

echo ""
echo "=========================================="
echo "All done. Results in ${SWEEP_DIR}/"
echo "=========================================="

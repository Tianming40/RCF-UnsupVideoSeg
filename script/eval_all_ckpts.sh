#!/bin/bash
# Sweep all checkpoints in a phase2 directory, evaluate on all 8 datasets.
#
# Usage:
#   bash script/eval_all_ckpts.sh <phase2_dir> [config] [gpu]
#
# Example:
#   bash script/eval_all_ckpts.sh saved/phase2_260604_080520
#   bash script/eval_all_ckpts.sh saved/phase2_260604_080520 configs/instrument/rcf_kfold_phase2.yaml 1

PHASE2_DIR=${1:?"Usage: $0 <phase2_dir> [config] [gpu]"}
CONFIG=${2:-configs/instrument/rcf_kfold_phase2.yaml}
GPU=${3:-0}
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
RESULTS="${PHASE2_DIR}/eval_results_${TIMESTAMP}.txt"

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd "$(dirname "$0")/.."

CKPTS=$(find "${PHASE2_DIR}" -maxdepth 2 -name "*.ckpt" | sort -V)
N=$(echo "${CKPTS}" | wc -l)

echo "==========================================" | tee "${RESULTS}"
echo "Evaluating ${N} checkpoints in ${PHASE2_DIR}" | tee -a "${RESULTS}"
echo "Config : ${CONFIG}" | tee -a "${RESULTS}"
echo "Dataset: trainval.txt (all 8 datasets)" | tee -a "${RESULTS}"
echo "Started: $(date)" | tee -a "${RESULTS}"
echo "==========================================" | tee -a "${RESULTS}"
echo "" | tee -a "${RESULTS}"

for CKPT in ${CKPTS}; do
    NAME=$(basename "${CKPT%.ckpt}")
    EVAL_DIR="${PHASE2_DIR}/evals/${TIMESTAMP}_${NAME}"
    mkdir -p "${EVAL_DIR}"

    echo "" | tee -a "${RESULTS}"
    echo "--- ${NAME} ---" | tee -a "${RESULTS}"
    echo "    eval dir: ${EVAL_DIR}"

    # Run two-pass test; save masks in per-checkpoint eval dir
    CUDA_VISIBLE_DEVICES=${GPU} python main.py "${CONFIG}" \
        --test \
        --test-override-pretrained "${CKPT}" \
        --opts \
            test_dataset_kwargs.split trainval.txt \
            eval_save True \
            eval_export False \
            export_all_seg False \
            allow_overwriting_checkpoints_dir True \
            saved_eval_dir_name "evals/${TIMESTAMP}_${NAME}/saved_eval" \
            eval_pos_th -1 \
        2>&1 | tee "${EVAL_DIR}/test.log" \
             | grep -E --line-buffered \
                 "Pass [12]|Set object channel|  instrument|test_miou|val_miou_frame_avg"

    # Extract key metrics into results file
    grep -E "Set object channel|  instrument|test_miou" \
        "${EVAL_DIR}/test.log" >> "${RESULTS}"
    echo "" | tee -a "${RESULTS}"
done

echo "==========================================" | tee -a "${RESULTS}"
echo "RANKING by test_miou (sequence-averaged)" | tee -a "${RESULTS}"
echo "==========================================" | tee -a "${RESULTS}"

# Extract (ckpt_name, miou) pairs and sort descending
python3 - "${RESULTS}" <<'EOF'
import sys, re

results_path = sys.argv[1]
text = open(results_path).read()

blocks = re.split(r"--- (.+?) ---", text)
pairs = []
for i in range(1, len(blocks), 2):
    name = blocks[i].strip()
    body = blocks[i+1]
    m = re.search(r"- INFO - test_miou:\s*([0-9.]+)", body)
    if m:
        pairs.append((name, float(m.group(1))))

pairs.sort(key=lambda x: x[1], reverse=True)
lines = []
for rank, (name, miou) in enumerate(pairs, 1):
    lines.append(f"  #{rank:2d}  {miou:.2f}%  {name}")
output = "\n".join(lines)
print(output)
# Append to results file
with open(results_path, "a") as f:
    f.write(output + "\n")
EOF

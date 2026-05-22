#!/bin/bash

#SBATCH --job-name=rcf_eval_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_rcf_eval_all_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

EVAL_DIR="saved/eval_all_ckpts"
mkdir -p "$EVAL_DIR"

RESULT_FILE="${EVAL_DIR}/results.txt"
echo "ckpt,test_miou" > $RESULT_FILE

for RUN_DIR in saved/saved_instrument_trainval_run*/; do
    for CKPT in "${RUN_DIR}"*.ckpt; do
        [[ "$CKPT" == *"last.ckpt" ]] && continue


        RUN_NAME=$(basename "$RUN_DIR")
        CKPT_NAME=$(basename "$CKPT" .ckpt)
        CKPT_EVAL_DIR="${EVAL_DIR}/${RUN_NAME}_${CKPT_NAME}"

        echo ""
        echo "=============================="
        echo "Evaluating: $CKPT"
        echo "Output dir: $CKPT_EVAL_DIR"
        echo "=============================="

        SCORE=$(CUDA_VISIBLE_DEVICES=0 python main.py \
            configs/instrument/rcf_instrument_trainval.yaml \
            --test \
            --test-override-pretrained "$CKPT" \
            --opts allow_overwriting_checkpoints_dir True \
                   checkpoints_dir "$CKPT_EVAL_DIR" \
                   eval_pos_th -1 \
                   test_dataset_kwargs.split trainval.txt \
            2>&1 | grep "test_miou: " | tail -1 | grep -oP "test_miou: \K[0-9.]+")

        SCORE=${SCORE:-0}
        echo "  Score: $SCORE"
        echo "${CKPT},${SCORE}" >> $RESULT_FILE
    done
done

echo ""
echo "=============================="
echo "All done! Results saved to $RESULT_FILE"
echo "Top 3 checkpoints:"
tail -n +2 $RESULT_FILE | sort -t',' -k2 -rn | head -3
echo "=============================="
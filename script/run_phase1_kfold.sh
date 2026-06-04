#!/bin/bash

#SBATCH --job-name=rcf_kfold_phase1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/script/slurm_out/slurm_kfold_phase1_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

# Run tag: YYMMDD_HHMMSS — used for all folder names and log files in this run
RUN_TAG=$(date +"%y%m%d_%H%M%S")

echo "=========================================="
echo "Phase 1: 4-fold cross-validation (mask_layer=5, epochs=80)"
echo "Train: fold{N}_train.txt (6 datasets, true held-out val)"
echo "Val:   fold{N}_val.txt   (2 datasets never seen during training)"
echo "Run tag: ${RUN_TAG}"
echo "Started at: $(date)"
echo "=========================================="

BEST_EPOCHS=()

for FOLD in 1 2 3 4; do
    RUN_DIR="saved/phase1_${RUN_TAG}_fold${FOLD}"

    echo ""
    echo "=========================================="
    echo "Fold ${FOLD} / 4"
    echo "Train: fold${FOLD}_train.txt  |  Val: fold${FOLD}_val.txt"
    echo "Checkpoint dir: ${RUN_DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 python main.py \
        configs/instrument/rcf_kfold_phase1.yaml \
        --opts \
            checkpoints_dir ${RUN_DIR} \
            train_dataset_kwargs.split fold${FOLD}_train.txt \
            test_dataset_kwargs.split fold${FOLD}_val.txt \
            allow_overwriting_checkpoints_dir True

    if [ $? -ne 0 ]; then
        echo "Fold ${FOLD} failed!"
        exit 1
    fi

    # Parse best epoch from checkpoint filename (epoch=N-...-val_miou_frame_avg=X.ckpt)
    BEST_EPOCH=$(python3 - <<'EOF'
import glob, re, os, sys
run_dir = sys.argv[1]
ckpts = glob.glob(f"{run_dir}/epoch=*.ckpt")
best_epoch, best_val = -1, -1.0
for c in ckpts:
    m = re.search(r"epoch=(\d+).*val_miou_frame_avg=([0-9.]+)", os.path.basename(c))
    if m:
        epoch, val = int(m.group(1)), float(m.group(2))
        if val > best_val:
            best_val, best_epoch = val, epoch
print(best_epoch)
EOF
    ${RUN_DIR})

    echo "Fold ${FOLD} → best epoch: ${BEST_EPOCH}"
    BEST_EPOCHS+=($BEST_EPOCH)
    echo "Fold ${FOLD} finished at: $(date)"
done

# Compute average best epoch across folds
AVG_EPOCH=$(python3 -c "
epochs = [${BEST_EPOCHS[@]}]
avg = round(sum(epochs) / len(epochs))
print(avg)
print(f'Per-fold best epochs: {epochs}', file=__import__(\"sys\").stderr)
")

echo ""
echo "=========================================="
echo "ALL 4 FOLDS FINISHED"
echo "Run tag          : ${RUN_TAG}"
printf "Per-fold dirs    : saved/phase1_%s_fold{1..4}\n" "${RUN_TAG}"
printf "Per-fold best epochs: %s\n" "${BEST_EPOCHS[*]}"
echo "Average best epoch   : ${AVG_EPOCH}"
echo ""
echo "Next step: run Phase 2 with this epoch count and same tag:"
echo "  bash script/run_phase2_final.sh ${AVG_EPOCH} ${RUN_TAG}"
echo "=========================================="

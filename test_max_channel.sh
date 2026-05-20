#!/bin/bash

#SBATCH --job-name=rcf_test_max_ch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_rcf_test_max_channel_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg


find_best_ckpt() {
    local dir=$1
    python -c "
import torch, glob, os
ckpts = [f for f in glob.glob('${dir}/*.ckpt') if 'last' not in f]
best_score = -1
best_path = ''
for ckpt in sorted(ckpts):
    try:
        d = torch.load(ckpt, map_location='cpu')
        for k, v in d.get('callbacks', {}).items():
            if 'ModelCheckpoint' in str(k):
                score = float(v.get('best_model_score') or -1)
                path  = v.get('best_model_path', '')
                if score > best_score and path and os.path.exists(path):
                    best_score = score
                    best_path  = path
    except Exception as e:
        pass
print(best_path if best_path else sorted(ckpts)[0])
"
}

for fold in 1 2 3 4; do
    echo "=========================================="
    echo "Testing Fold $fold  (max channel, hard argmax)"
    echo "Started at: $(date)"
    echo "=========================================="

    CKPT=$(find_best_ckpt "saved/saved_instrument_fold${fold}")
    echo "Using checkpoint: $CKPT"

    # eval_pos_th=-1 → hard argmax
    # object_channel  → main.py  None → max channel
    CUDA_VISIBLE_DEVICES=0 python main.py \
        configs/instrument/rcf_instrument_fold${fold}.yaml \
        --test \
        --test-override-pretrained "$CKPT" \
        --opts allow_overwriting_checkpoints_dir True eval_pos_th -1

    echo "Fold $fold done at: $(date)"
    echo ""
done

echo "All folds finished at: $(date)"

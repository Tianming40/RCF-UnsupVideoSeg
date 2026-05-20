#!/bin/bash

#SBATCH --job-name=rcf_instrument_cv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_rcf_instrument_cv_%j.out

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg

for fold in 1 2 3 4; do
    echo "=========================================="
    echo "Training Fold $fold"
    echo "Started at: $(date)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python main.py configs/instrument/rcf_instrument_fold${fold}.yaml
    
    if [ $? -eq 0 ]; then
        echo "Fold $fold completed at: $(date)"
    else
        echo "Fold $fold failed!"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "All 4 folds complete!"
echo "Finished at: $(date)"
echo "=========================================="
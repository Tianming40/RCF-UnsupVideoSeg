#!/bin/bash
# Generate RAFT optical flows for CMC_grasp0_deinterlaced.

source /home/tianming/anaconda3/etc/profile.d/conda.sh
conda activate rcf

cd /media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/RAFT

CUDA_VISIBLE_DEVICES=0 python generate_flows_cmc.py \
    --model models/raft-things.pth \
    --data_root /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced

echo "Done. Flows saved to CMC_grasp0_deinterlaced/Flows_NewCT/ and BackwardFlows_NewCT/"

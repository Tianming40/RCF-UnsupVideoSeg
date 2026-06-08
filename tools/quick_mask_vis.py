#!/usr/bin/env python3
"""


Usage:
  python tools/quick_mask_vis.py \
    --config  configs/instrument/rcf_cmc_dino_phase1.yaml \
    --ckpt    saved/cmc_dino_phase1_260605_143205/epoch=7-step=1800.ckpt \
    --output  analysis/quick_mask_vis \
    --split   trainval.txt \
    --n       20
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
_models_pkg.RCFDinoModel = RCFDinoModel

from tools.maa_union_inference import load_config, load_checkpoint, build_probe_dataset, to_device

import argparse as _ap
from models.rcf_model import RCFModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DINO_ONLY = frozenset({'w_dino','dino_checkpoint','dino_arch','dino_patch_size','dino_input_size'})


def build_base_model(cfg, output_dir):
    cfg_c = copy.deepcopy(cfg)
    for k in _DINO_ONLY:
        cfg_c.get('model_kwargs', {}).pop(k, None)
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg_c.get('eval_pos_th', 0.35),
        object_channel=None, log_interval=9999,
    )
    return RCFModel(args=fake_args, **copy.deepcopy(cfg_c['model_kwargs']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--ckpt',    required=True)
    p.add_argument('--output',  required=True)
    p.add_argument('--split',   default='trainval.txt')
    p.add_argument('--flow_suffix', default='_NewCT')
    p.add_argument('--n',       type=int, default=20, help='number of frames to visualize')
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model = build_base_model(cfg, args.output)
    model = load_checkpoint(model, args.ckpt)
    model = model.to(DEVICE)
    model.eval()

    dataset = build_probe_dataset(cfg, split_override=args.split,
                                  flow_suffix=args.flow_suffix)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    C = model.num_classes   # 5

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=args.n)):
            if i >= args.n:
                break

            batch = to_device(batch)
            imgs  = torch.stack(batch['imgs'], dim=1)   # [1, F, 3, H, W]

            # first frame only
            img0  = imgs[:, 0]                           # [1, 3, H, W]
            masks = model.forward_eval(
                imgs, batch['seq_ids'], batch['seq_names'], batch['paths'])  # [1, C, fH, fW]

            H, W = img0.shape[2], img0.shape[3]

            # --- original ---
            img_01 = ((img0.cpu() + 2.0) / 4.0).clamp(0, 1)   # [1, 3, H, W]

            # --- B ---
            masks_up = F.interpolate(masks.cpu(), (H, W),
                                     mode='bilinear', align_corners=False)  # [1, C, H, W]
            mask_imgs = []
            for c in range(C):
                m = masks_up[0, c]                   # [H, W]  in [0,1]
                m_rgb = m.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,H,W]
                mask_imgs.append(m_rgb)

            # ---  [, ch0, ch1, ch2, ch3, ch4] ---
            row = torch.cat([img_01] + mask_imgs, dim=0)  # [6, 3, H, W]

            seq   = batch['seq_names'][0]
            fname = batch['paths'][0][0].split('/')[-1][:-4]
            save_path = out / f'{i+1:03d}_{seq}_{fname}.jpg'

            # nrow=6 
            torchvision.utils.save_image(row, str(save_path), nrow=6, padding=4, pad_value=0.5)

    print(f'Done → {out}  ({min(i+1, args.n)} frames)')
    print('order: or | ch0 | ch1 | ch2 | ch3 | ch4')


if __name__ == '__main__':
    main()

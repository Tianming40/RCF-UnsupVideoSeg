#!/usr/bin/env python3
"""
CMC visualization inference.

Saves a grid of 8 images per frame:
  1. Original image
  2-6. Channel 0-4 soft masks
  7. Union binary mask (channels 0, 2, 3)
  8. Green overlay (union mask on original)

Usage:
  python tools/cmc_vis_inference.py \
    --config  configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml \
    --ckpt    saved/cmc_grasp10_finetune_v2b_260604_120527/last.ckpt \
    --output  saved/cmc_vis_last \
    --split   ImageSets/val.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "SemanticConstraintsAndMAA"))

# Register V2 into rcf_model namespace
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

from tools.maa_union_inference import (
    load_config, build_model, load_checkpoint,
    build_dataset, to_device,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_green_overlay(orig_r: torch.Tensor, union_bin: torch.Tensor) -> torch.Tensor:
    """
    Args:
        orig_r:    [1, 3, H, W]  original image in [0, 1]
        union_bin: [H, W]        binary union mask
    Returns:
        overlay:   [1, 3, H, W]  green overlay
    """
    overlay = orig_r.clone()
    m = union_bin.bool()
    # R: dim masked region
    overlay[0, 0][m] = overlay[0, 0][m] * 0.3
    # G: boost masked region (adapted from maa_inference_ensemble_areaNorm.py)
    overlay[0, 1][m] = (overlay[0, 1][m] * 0.5 + 180 / 255.).clamp(0, 1)
    # B: dim masked region
    overlay[0, 2][m] = overlay[0, 2][m] * 0.3
    return overlay


def run_vis_inference(model, dataset, out_dir: str,
                      union_channels=(0, 2, 3),
                      eval_pos_th: float = 0.35,
                      workers: int = 0):
    out_dir = Path(out_dir)
    save_dir = out_dir / "saved_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_dir_eval = str(save_dir)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=False)

    print(f"Running inference on {len(dataset)} frames → {save_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = to_device(batch)
            imgs = torch.stack(batch["imgs"], dim=1)  # [1, 1, 3, H, W]

            # forward_eval builds base grid (original + 5 channel masks)
            # and stores it in model._pending_eval_viz
            pred_masks = model.forward_eval(
                imgs, batch["seq_ids"], batch["seq_names"], batch["paths"])
            # pred_masks: [1, C, fh, fw]

            p = model._pending_eval_viz
            if p is None:
                continue

            row_h, row_w = p["row_h"], p["row_w"]
            base = p["tosave"]   # [1, 3, 6*row_h, row_w]
            device = base.device

            # Soft masks resized to row resolution
            soft = F.softmax(pred_masks, dim=1)          # [1, C, fh, fw]
            soft_r = F.interpolate(soft, (row_h, row_w),
                                   mode="bilinear", align_corners=False)

            # Union binary mask
            union_prob = soft_r[0, union_channels[0]].clone()
            for ch in union_channels[1:]:
                union_prob = torch.max(union_prob, soft_r[0, ch])
            union_bin = (union_prob > eval_pos_th).float()   # [row_h, row_w]

            # Union row: grayscale → 3 channels [1, 3, row_h, row_w]
            union_row = union_bin[None, None].repeat(1, 3, 1, 1).to(device)

            # Original image at row resolution (model stores it at row scale already)
            orig = imgs[0, 0]  # [3, H, W]
            orig_r = F.interpolate(orig[None], (row_h, row_w),
                                   mode="bilinear", align_corners=False)
            orig_r = ((orig_r + 2.0) / 4.0).clamp(0, 1)

            # Green overlay row [1, 3, row_h, row_w]
            overlay_row = make_green_overlay(orig_r.cpu(), union_bin.cpu()).to(device)

            # Final 8-row grid
            tosave = torch.cat([base, union_row, overlay_row], dim=2)

            # Clear pending and save
            model._pending_eval_viz = None
            model.save_eval_visualizations(
                tosave, p["paths"], p["seq_ids"], p["seq_names"],
                train_iter=model.train_iter)

    print(f"Done → {save_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--split",         default="ImageSets/val.txt")
    p.add_argument("--union_channels",type=int, nargs="+", default=[0, 2, 3])
    p.add_argument("--pos_th",        type=float, default=0.35)
    p.add_argument("--workers",       type=int,   default=0)
    args = p.parse_args()

    cfg   = load_config(args.config)
    cfg["model_kwargs"]["allow_mask_resize"] = True
    model = build_model(cfg, output_dir=args.output)
    model = load_checkpoint(model, args.ckpt)

    dataset = build_dataset(cfg, split_override=args.split)
    run_vis_inference(model, dataset, args.output,
                      union_channels=tuple(args.union_channels),
                      eval_pos_th=args.pos_th,
                      workers=args.workers)


if __name__ == "__main__":
    main()

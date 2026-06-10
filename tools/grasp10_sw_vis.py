#!/usr/bin/env python3
"""
Sliding-window visualization for grasp10 (RCFDinoModel).

Replicates the forward_eval grid format (orig | ch0 | ch1 | ch2 | ch3 | ch4
concatenated vertically) but uses sliding-window inference so BN always sees
384×384 crops.  Fills the gap that main.py leaves: eval_save is disabled
whenever use_sliding_window=True.

Usage:
    python tools/grasp10_sw_vis.py \\
        --config  configs/instrument/rcf_cmc_grasp10_ft_v9_fulltrain_ft.yaml \\
        --ckpt    saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt \\
        --output  saved/grasp10_sw_vis \\
        --split   ImageSets/trainval_single.txt \\
        --gpu     1
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Register RCFDinoModel ──────────────────────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
_models_pkg.RCFDinoModel = RCFDinoModel

from tools.maa_union_inference import load_config, load_checkpoint, build_dataset, to_device


# ── Sliding window (mirrors main.py _sliding_window_eval) ─────────────────────

def sliding_window_eval(model, imgs, window_size=384, stride=192):
    """
    imgs : [B, im_num, 3, H, W]  (normalized, same as training)
    Returns [B, C, H, W]  soft-max probabilities at full image resolution.
    """
    B, im_num, C3, H, W = imgs.shape
    mask_layer = model.mask_layer

    pred_accum = torch.zeros(B, mask_layer, H, W, device=imgs.device)
    count_map  = torch.zeros(1, 1, H, W, device=imgs.device)

    def _positions(length):
        if length <= window_size:
            return [0]
        pts = list(range(0, length - window_size, stride))
        if not pts or pts[-1] + window_size < length:
            pts.append(length - window_size)
        return sorted(set(pts))

    for y in _positions(H):
        for xp in _positions(W):
            y2 = min(y + window_size, H)
            x2 = min(xp + window_size, W)
            crop = imgs[:, :, :, y:y2, xp:x2]
            ch, cw = crop.shape[-2], crop.shape[-1]
            if ch < window_size or cw < window_size:
                crop = F.pad(crop, (0, window_size - cw, 0, window_size - ch))

            img3 = crop.view(B * im_num, C3, window_size, window_size)
            feat = model.extract_feat(img3, model.backbone2)
            pred = model._decode_head_forward(feat, model.decode_head2)
            pred = pred.view(B, im_num, mask_layer, *pred.shape[-2:])
            pred = F.softmax(pred, dim=2)[:, 0]            # [B, C, fH, fW]
            pred = F.interpolate(pred, size=(window_size, window_size),
                                 mode='bilinear', align_corners=False)
            pred_accum[:, :, y:y2, xp:x2] += pred[:, :, :ch, :cw]
            count_map[:, :, y:y2, xp:x2] += 1

    return pred_accum / count_map.clamp(min=1)   # [B, C, H, W]


# ── Visualization (mirrors rcf_model.forward_eval grid format) ─────────────────

def make_grid(orig_normalized, pred_masks, display_h=None):
    """
    orig_normalized : [3, H, W]  (normalized, range ~[-2, 2])
    pred_masks      : [C, H, W]  (softmax probs, full resolution)
    display_h       : resize rows to this height (None = keep original)

    Returns [3, (C+1)*dH, dW]  — same stacking as forward_eval's tosave.
    """
    # Denormalize image to [0, 1]
    orig_01 = ((orig_normalized.float().cpu() + 2.0) / 4.0).clamp(0, 1)

    H, W = orig_01.shape[1], orig_01.shape[2]
    if display_h is not None and display_h != H:
        scale = display_h / H
        dH, dW = display_h, int(W * scale)
        orig_01 = F.interpolate(orig_01.unsqueeze(0), (dH, dW),
                                mode='bilinear', align_corners=False)[0]
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), (dH, dW),
                                   mode='bilinear', align_corners=False)[0]
    else:
        dH, dW = H, W

    rows = [orig_01]
    for c in range(pred_masks.shape[0]):
        rows.append(pred_masks[c:c+1].cpu().repeat(3, 1, 1))

    return torch.cat(rows, dim=1)   # [3, (C+1)*dH, dW]


# ── Build RCFDinoModel (not RCFModel) ─────────────────────────────────────────

def build_dino_model(cfg, output_dir="/tmp"):
    import argparse as _ap, copy
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    kwargs = copy.deepcopy(cfg["model_kwargs"])
    kwargs["allow_mask_resize"] = True
    return RCFDinoModel(args=fake_args, **kwargs)


# ── Main inference loop ────────────────────────────────────────────────────────

def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg     = load_config(args.config)
    out_dir = Path(args.output)
    vis_dir = out_dir / "saved_eval"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = build_dino_model(cfg, output_dir=str(out_dir))
    # load_checkpoint strips the "model." prefix from PL state_dict
    model = load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    dataset = build_dataset(cfg, split_override=args.split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    sw_size   = cfg.get("sliding_window_size",   384)
    sw_stride = cfg.get("sliding_window_stride", 192)
    mask_layer = cfg["model_kwargs"]["mask_layer"]

    print(f"Sliding window: size={sw_size}, stride={sw_stride}")
    print(f"Frames: {len(dataset)}  →  {vis_dir}")

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                         [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
                         if isinstance(v, list) else v)
                     for k, v in batch.items()}

            imgs = torch.stack(batch["imgs"], dim=1)   # [1, 1, 3, H, W]

            pred = sliding_window_eval(model, imgs, sw_size, sw_stride)  # [1, C, H, W]

            grid = make_grid(imgs[0, 0], pred[0], display_h=args.display_h)  # [3, rows*H, W]

            # Same filename convention as forward_eval / save_eval_visualizations
            seq_name = batch["seq_names"][0]
            seq_id   = batch["seq_ids"][0] if isinstance(batch["seq_ids"][0], str) \
                       else str(int(batch["seq_ids"][0]))
            frame_id = batch["paths"][0][0].split("/")[-1][:-4]
            fn = vis_dir / f"eval_{seq_name}_{seq_id}_{frame_id}_0000000.jpg"

            torchvision.utils.save_image(grid.unsqueeze(0), str(fn))

    print(f"Done. {len(dataset)} frames saved to {vis_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--split",     default=None,
                   help="Override split (default: test_dataset_kwargs.split from config)")
    p.add_argument("--gpu",       type=int, default=1)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--display_h", type=int, default=None,
                   help="Resize each row to this height for the grid (default: full res)")
    run(p.parse_args())

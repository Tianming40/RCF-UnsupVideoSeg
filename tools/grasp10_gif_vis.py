#!/usr/bin/env python3
"""
Grasp10 sliding-window visualization — horizontal grid format.

For each frame saves a 1-row grid to {output}/gif/:
  orig | ch0 | ch1 | ch2 | ch3 | ch4 | union-overlay

All channel masks are Guided-Filter smoothed then binary-thresholded.

Usage:
    python tools/grasp10_gif_vis.py \
        --config  configs/instrument/rcf_cmc_grasp10_ft_v9_fulltrain_ft.yaml \
        --ckpt    saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt \
        --output  saved/grasp10_gif_vis \
        --split   ImageSets/trainval_single.txt \
        --gpu     1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
_models_pkg.RCFDinoModel = RCFDinoModel

from tools.maa_union_inference import load_config, load_checkpoint, build_dataset, to_device


# ── Sliding window (mirrors main.py _sliding_window_eval) ────────────────────

def sliding_window_eval(model, imgs, window_size=384, stride=192):
    """imgs: [B, im_num, 3, H, W] → returns [B, C, H, W] soft probs at full res."""
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
            y2, x2 = min(y + window_size, H), min(xp + window_size, W)
            crop = imgs[:, :, :, y:y2, xp:x2]
            ch, cw = crop.shape[-2], crop.shape[-1]
            if ch < window_size or cw < window_size:
                crop = F.pad(crop, (0, window_size - cw, 0, window_size - ch))
            img3 = crop.view(B * im_num, C3, window_size, window_size)
            feat = model.extract_feat(img3, model.backbone2)
            pred = model._decode_head_forward(feat, model.decode_head2)
            pred = pred.view(B, im_num, mask_layer, *pred.shape[-2:])
            pred = F.softmax(pred, dim=2)[:, 0]
            pred = F.interpolate(pred, size=(window_size, window_size),
                                 mode='bilinear', align_corners=False)
            pred_accum[:, :, y:y2, xp:x2] += pred[:, :, :ch, :cw]
            count_map[:, :, y:y2, xp:x2] += 1

    return pred_accum / count_map.clamp(min=1)


# ── Guided filter ─────────────────────────────────────────────────────────────

def guided_filter(guide_rgb, src, r=16, eps=1e-2):
    """Smooth src using guide_rgb as edge guide. guide_rgb/src: [B, *, H, W]."""
    I = (0.299 * guide_rgb[:, 0:1] + 0.587 * guide_rgb[:, 1:2]
         + 0.114 * guide_rgb[:, 2:3])
    k = 2 * r + 1

    def mean_f(x):
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='reflect'), k, stride=1, padding=0)

    mean_I  = mean_f(I)
    mean_p  = mean_f(src)
    cov_Ip  = mean_f(I * src) - mean_I * mean_p
    var_I   = mean_f(I * I)   - mean_I * mean_I
    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I
    return (mean_f(a) * I + mean_f(b)).clamp(0, 1)


# ── Color overlay (same as tissue_vis_inference) ──────────────────────────────

def color_overlay(img_01, mask, color):
    """img_01: [3,H,W], mask: [H,W] in [0,1], color: (R,G,B) in [0,1]."""
    out = img_01.clone()
    m = mask.unsqueeze(0)
    c = torch.tensor(color, device=img_01.device).view(3, 1, 1)
    return (out * (1 - m * 0.6) + c * (m * 0.6)).clamp(0, 1)


# ── Build RCFDinoModel ────────────────────────────────────────────────────────

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


# ── Main inference loop ───────────────────────────────────────────────────────

def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg     = load_config(args.config)
    out_dir = Path(args.output)
    gif_dir = out_dir / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)

    model = build_dino_model(cfg, output_dir=str(out_dir))
    model = load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    sw_size   = getattr(args, 'sw_size',   cfg.get("sliding_window_size",   384))
    sw_stride = getattr(args, 'sw_stride', cfg.get("sliding_window_stride", 192))
    mask_layer = cfg["model_kwargs"]["mask_layer"]

    if args.data_path:
        cfg["data_path"] = args.data_path
    dataset = build_dataset(cfg, split_override=args.split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    print(f"Sliding window: size={sw_size}, stride={sw_stride}")
    print(f"Red channels: {args.red_channels}  Blue channels: {args.blue_channels}  th: {args.pos_th}")
    print(f"Frames: {len(dataset)}  →  {gif_dir}")

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = to_device(batch)
            imgs = torch.stack(batch["imgs"], dim=1)   # [1, 1, 3, H, W]

            pred = sliding_window_eval(model, imgs, sw_size, sw_stride)  # [1, C, H, W]

            orig_01 = ((imgs[0, 0] + 2.0) / 4.0).clamp(0, 1)            # [3, H, W]
            H, W = orig_01.shape[1], orig_01.shape[2]

            # Guided filter on full-res soft probs
            pred_raw    = pred[0]                                          # [C, H, W] soft
            pred_smooth = guided_filter(orig_01.unsqueeze(0),
                                        pred, r=args.gf_r, eps=args.gf_eps)[0]  # [C, H, W]

            th          = args.pos_th
            red_chs     = list(args.red_channels)
            blue_chs    = list(args.blue_channels)

            def _union_bin(probs, channels):
                p = probs[channels[0]]
                for c in channels[1:]:
                    p = torch.max(p, probs[c])
                return (p > th).float()

            def _build_row(probs, binary):
                """9 cols: orig | ch0..ch4 | red_ov | blue_ov | dual_ov"""
                row = [orig_01]
                for c in range(mask_layer):
                    row.append(probs[c:c+1].clamp(0, 1).repeat(3, 1, 1) if not binary
                               else (probs[c:c+1] > th).float().repeat(3, 1, 1))
                r_bin = _union_bin(probs, red_chs)
                b_bin = _union_bin(probs, blue_chs)
                row.append(color_overlay(orig_01, r_bin, (1.0, 0.2, 0.2)))
                row.append(color_overlay(orig_01, b_bin, (0.3, 0.6, 1.0)))
                dual = color_overlay(orig_01, r_bin, (1.0, 0.2, 0.2))
                dual = color_overlay(dual,    b_bin, (0.3, 0.6, 1.0))
                row.append(dual)
                return row

            row1 = _build_row(pred_raw,    binary=False)  # raw soft prob
            row2 = _build_row(pred_smooth, binary=True)   # GF binary

            # 2 rows × 9 cols via make_grid(nrow=9)
            all_imgs = torch.stack(row1 + row2, dim=0)    # [18, 3, H, W]
            grid = torchvision.utils.make_grid(all_imgs, nrow=len(row1), padding=2)

            seq_name   = batch["seq_names"][0]
            frame_name = batch["paths"][0][0].split("/")[-1][:-4]
            save_path  = gif_dir / f"{seq_name}_{frame_name}.jpg"
            torchvision.utils.save_image(grid, str(save_path))

    print(f"Done. {len(dataset)} frames → {gif_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--split",   default=None)
    p.add_argument("--gpu",     type=int,   default=1)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--pos_th",  type=float, default=0.35)
    p.add_argument("--gf_r",    type=int,   default=16,   help="Guided filter radius")
    p.add_argument("--gf_eps",  type=float, default=1e-2, help="Guided filter edge threshold")
    p.add_argument("--data_path", default=None, help="覆盖 config 里的 data_path")
    p.add_argument("--red_channels",  type=int, nargs="+", default=[1],
                   help="Channels for red overlay, e.g. --red_channels 1 2")
    p.add_argument("--blue_channels", type=int, nargs="+", default=[2, 3],
                   help="Channels for blue overlay, e.g. --blue_channels 3 4")
    run(p.parse_args())

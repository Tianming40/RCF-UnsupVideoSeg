#!/usr/bin/env python3
"""
grasp0_mask_export.py

Sliding-window inference on grasp0 sequences → saves all masks in multiple formats.

Per sequence outputs:
  npy/        <seq>.npy          [C, H, W] float16  raw soft prob
              <seq>_gf.npy       [C, H, W] float16  guided-filter soft prob
  ch/<c>/     <seq>.png          grayscale raw soft prob, channel c  (0-4)
  ch_gf/<c>/  <seq>.png          grayscale GF soft prob, channel c
  overlay/    <seq>.png          colour overlay (all channels) on original frame
  gif/        <seq>_<frame>.jpg  2-row × 9-col debug grid (raw | GF binary)

Usage:
    python tools/grasp0_mask_export.py \\
        --config configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml \\
        --ckpt   saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt \\
        --output saved/grasp0_mask_export \\
        --split  ImageSets/trainval_single.txt \\
        --gpu    1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Model registration (same pattern as main_grasp0.py) ──────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
_models_pkg.RCFDinoModel = RCFDinoModel

from models.rcf_soft_tissue_model import RCFSoftTissueModel
_models_pkg.RCFSoftTissueModel = RCFSoftTissueModel

from tools.maa_union_inference import load_config, load_checkpoint, build_dataset, to_device
from tools.grasp10_gif_vis import sliding_window_eval, guided_filter, color_overlay


def build_model(cfg, output_dir="/tmp"):
    """Instantiate the model class specified in cfg['model_cls']."""
    import argparse as _ap, copy
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    kwargs = copy.deepcopy(cfg["model_kwargs"])
    kwargs["allow_mask_resize"] = True
    kwargs["w_distill"] = 0.0   # skip teacher build during inference

    model_cls_name = cfg.get("model_cls", "RCFDinoModel")
    model_cls = getattr(_models_pkg, model_cls_name)
    return model_cls(args=fake_args, **kwargs)

# ── Per-channel overlay colours ───────────────────────────────────────────────
CHANNEL_COLORS = [
    (0.2, 0.8, 0.2),   # ch0  green
    (1.0, 0.2, 0.2),   # ch1  red  — instrument
    (1.0, 0.9, 0.1),   # ch2  yellow — soft tissue target
    (0.2, 0.4, 1.0),   # ch3  blue
    (0.8, 0.2, 0.8),   # ch4  purple
]


def _ensure(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg     = load_config(args.config)
    out_dir = Path(args.output)

    # ── Output sub-directories ────────────────────────────────────────────────
    npy_dir     = out_dir / "npy"
    gif_dir     = out_dir / "gif"
    overlay_dir = out_dir / "overlay"
    sw_size   = args.sw_size   or cfg.get("sliding_window_size",   384)
    sw_stride = args.sw_stride or cfg.get("sliding_window_stride", 192)
    mask_layer  = cfg["model_kwargs"]["mask_layer"]

    ch_dirs    = [out_dir / "ch"    / str(c) for c in range(mask_layer)]
    ch_gf_dirs = [out_dir / "ch_gf" / str(c) for c in range(mask_layer)]
    _ensure(npy_dir, gif_dir, overlay_dir, *ch_dirs, *ch_gf_dirs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, output_dir=str(out_dir))
    model = load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    if args.data_path:
        cfg["data_path"] = args.data_path
    dataset = build_dataset(cfg, split_override=args.split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    print(f"Sliding window: size={sw_size}, stride={sw_stride}")
    print(f"Channels: {mask_layer}   GF: r={args.gf_r} eps={args.gf_eps}   th={args.pos_th}")
    print(f"Frames: {len(dataset)}  →  {out_dir}")

    with torch.no_grad():
        for batch in tqdm(loader):
            batch     = to_device(batch)
            imgs      = torch.stack(batch["imgs"], dim=1)   # [1, 1, 3, H, W]
            seq_name  = batch["seq_names"][0]
            frame_name = batch["paths"][0][0].split("/")[-1][:-4]

            # ── Inference ─────────────────────────────────────────────────────
            pred_raw = sliding_window_eval(model, imgs, sw_size, sw_stride)  # [1, C, H, W]

            orig_01  = ((imgs[0, 0] + 2.0) / 4.0).clamp(0, 1)              # [3, H, W]

            pred_gf  = guided_filter(orig_01.unsqueeze(0), pred_raw,
                                     r=args.gf_r, eps=args.gf_eps)          # [1, C, H, W]

            raw = pred_raw[0]   # [C, H, W]
            gf  = pred_gf[0]   # [C, H, W]

            # ── npy (float16) ─────────────────────────────────────────────────
            np.save(str(npy_dir / f"{seq_name}.npy"),
                    raw.cpu().numpy().astype(np.float16))
            np.save(str(npy_dir / f"{seq_name}_gf.npy"),
                    gf.cpu().numpy().astype(np.float16))

            # ── per-channel PNG (grayscale, raw + GF) ─────────────────────────
            for c in range(mask_layer):
                torchvision.utils.save_image(
                    raw[c:c+1], str(ch_dirs[c] / f"{seq_name}.png"))
                torchvision.utils.save_image(
                    gf[c:c+1].clamp(0, 1), str(ch_gf_dirs[c] / f"{seq_name}.png"))

            # ── colour overlay (all GF channels on original) ──────────────────
            ovl = orig_01.clone()
            for c in range(min(mask_layer, len(CHANNEL_COLORS))):
                mask_c = (gf[c] > args.pos_th).float()
                ovl = color_overlay(ovl, mask_c, CHANNEL_COLORS[c])
            torchvision.utils.save_image(ovl, str(overlay_dir / f"{seq_name}.png"))

            # ── 2-row × 9-col debug grid (raw soft | GF binary) ──────────────
            def _union_bin(probs, channels):
                p = probs[channels[0]]
                for c in channels[1:]:
                    p = torch.max(p, probs[c])
                return (p > args.pos_th).float()

            def _build_row(probs, binary):
                row = [orig_01]
                for c in range(mask_layer):
                    row.append(probs[c:c+1].clamp(0, 1).repeat(3, 1, 1) if not binary
                               else (probs[c:c+1] > args.pos_th).float().repeat(3, 1, 1))
                r_bin = _union_bin(probs, args.red_channels)
                b_bin = _union_bin(probs, args.blue_channels)
                row.append(color_overlay(orig_01, r_bin, (1.0, 0.2, 0.2)))
                row.append(color_overlay(orig_01, b_bin, (0.3, 0.6, 1.0)))
                dual = color_overlay(orig_01, r_bin, (1.0, 0.2, 0.2))
                dual = color_overlay(dual,    b_bin, (0.3, 0.6, 1.0))
                row.append(dual)
                return row

            row1 = _build_row(raw, binary=False)
            row2 = _build_row(gf,  binary=True)
            grid = torchvision.utils.make_grid(
                torch.stack(row1 + row2, dim=0), nrow=len(row1), padding=2)
            torchvision.utils.save_image(
                grid, str(gif_dir / f"{seq_name}_{frame_name}.jpg"))

    print(f"\nDone. Outputs in {out_dir}/")
    print(f"  npy/          raw + GF soft probs  [C,H,W] float16")
    print(f"  ch/0..{mask_layer-1}/      per-channel raw grayscale PNG")
    print(f"  ch_gf/0..{mask_layer-1}/   per-channel GF grayscale PNG")
    print(f"  overlay/      colour overlay PNG")
    print(f"  gif/          2×9 debug grid JPG")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--split",   default=None)
    p.add_argument("--gpu",     type=int,   default=1)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--pos_th",  type=float, default=0.35)
    p.add_argument("--gf_r",    type=int,   default=16)
    p.add_argument("--gf_eps",  type=float, default=1e-2)
    p.add_argument("--data_path", default=None)
    p.add_argument("--sw_size",   type=int,   default=None)
    p.add_argument("--sw_stride", type=int,   default=None)
    p.add_argument("--red_channels",  type=int, nargs="+", default=[1],
                   help="Channels for red overlay in debug grid")
    p.add_argument("--blue_channels", type=int, nargs="+", default=[2, 3],
                   help="Channels for blue overlay in debug grid")
    run(p.parse_args())

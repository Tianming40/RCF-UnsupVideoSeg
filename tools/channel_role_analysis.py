#!/usr/bin/env python3
"""
Channel Role Analysis for RCF/Instrument model.



Usage:
  python tools/channel_role_analysis.py \
    --config configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml \
    --ckpt   saved/cmc_grasp10_finetune_v2b_260604_120527/last.ckpt \
    --output analysis_channel_roles \
    --n_samples 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "SemanticConstraintsAndMAA"))

# Register V2 into rcf_model namespace (must be before any rcf_model import)
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

from tools.maa_union_inference import (
    load_config, build_model, load_checkpoint,
    build_probe_dataset, to_device, RCFFeatureExtractor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Flow analysis utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_convergence_map(flow: torch.Tensor) -> torch.Tensor:
    """
    flow: [B, 2, H, W]
    Returns convergence map [B, H, W] in [0, 1] (per-sample normalized).
    High value = flow converges here (negative divergence).
    """
    fx = flow[:, 0]
    fy = flow[:, 1]
    fx_p = F.pad(fx.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
    fy_p = F.pad(fy.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
    dfx_dx = (fx_p[:, 1:-1, 2:] - fx_p[:, 1:-1, :-2]) / 2.0
    dfy_dy = (fy_p[:, 2:, 1:-1] - fy_p[:, :-2, 1:-1]) / 2.0
    div = dfx_dx + dfy_dy                         # [B, H, W]
    conv = (-div).clamp(min=0)
    B = conv.shape[0]
    max_v = conv.view(B, -1).max(dim=1).values    # [B]
    return conv / (max_v.view(B, 1, 1) + 1e-6)


def per_channel_stats(masks: torch.Tensor,
                      flow: torch.Tensor,
                      residual_fw: torch.Tensor,
                      pred_div_coeff: float = 10.,
                      residual_scale: float = 10.) -> dict:
    """
    masks:        [B, C, H, W]  softmax masks (already at flow resolution)
    flow:         [B, 2, H, W]  RAFT flow at mask resolution
    residual_fw:  [B, 2*C, H, W] raw decode_head3 output (fw direction)

    Returns dict with keys → tensors of shape [C] (cpu).
    """
    B, C, H, W = masks.shape
    flow_mag = flow.norm(dim=1, keepdim=False)   # [B, H, W]
    angle    = torch.atan2(flow[:, 1], flow[:, 0])  # [B, H, W]
    conv_map = compute_convergence_map(flow)       # [B, H, W]

    # Per-channel residual magnitude (tanh-scaled, same as model)
    # residual_fw: [B, 2*C, H, W] → [B, 2, C, H, W]
    res_per_ch = torch.tanh(residual_fw.unflatten(1, (2, C)) / pred_div_coeff) \
                 * residual_scale                  # [B, 2, C, H, W]
    res_mag_per_ch = res_per_ch.norm(dim=1)        # [B, C, H, W]

    stats = {
        'mean_flow_mag':      torch.zeros(C),
        'flow_dir_var':       torch.zeros(C),
        'mean_residual_mag':  torch.zeros(C),
        'convergence_score':  torch.zeros(C),
        'mean_flow_x':        torch.zeros(C),
        'mean_flow_y':        torch.zeros(C),
        'mask_area':          torch.zeros(C),
    }

    for c in range(C):
        m   = masks[:, c]                               # [B, H, W]
        w   = m.sum(dim=(1, 2)) + 1e-6                  # [B]

        # 1. Weighted mean flow magnitude
        stats['mean_flow_mag'][c]     = ((flow_mag * m).sum(dim=(1,2)) / w).mean()

        # 2. Weighted flow direction variance (circular)
        mu_angle = ((angle * m).sum(dim=(1,2)) / w).view(B,1,1)
        diff     = angle - mu_angle
        diff     = torch.remainder(diff + torch.pi, 2*torch.pi) - torch.pi
        stats['flow_dir_var'][c]      = ((diff**2 * m).sum(dim=(1,2)) / w).mean()

        # 3. Weighted residual magnitude
        stats['mean_residual_mag'][c] = ((res_mag_per_ch[:, c] * m).sum(dim=(1,2)) / w).mean()

        # 4. Convergence score
        stats['convergence_score'][c] = ((conv_map * m).sum(dim=(1,2)) / w).mean()

        # 5. Net flow direction
        stats['mean_flow_x'][c]       = ((flow[:, 0] * m).sum(dim=(1,2)) / w).mean()
        stats['mean_flow_y'][c]       = ((flow[:, 1] * m).sum(dim=(1,2)) / w).mean()

        # 6. Mask area
        stats['mask_area'][c]         = m.mean()

    return stats, conv_map


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def colorize_heatmap(t: np.ndarray, cmap='plasma') -> np.ndarray:
    """[H, W] float [0,1] → [3, H, W] float [0,1]"""
    cm   = plt.get_cmap(cmap)
    rgba = cm(t)
    return rgba[:, :, :3].transpose(2, 0, 1).astype(np.float32)


def flow_to_rgb(flow_np: np.ndarray) -> np.ndarray:
    """[2, H, W] → [3, H, W] float [0,1]"""
    import flow_vis
    rgb = flow_vis.flow_to_color(flow_np.transpose(1, 2, 0), convert_to_bgr=False)
    return rgb.astype(np.float32).transpose(2, 0, 1) / 255.0


def save_sample_vis(img, masks, flow, conv_map, save_path,
                    instrument_channels, tissue_channel):
    """
    Saves a multi-row grid for one sample:
      Row 0: original image
      Row 1: RAFT flow (color-coded)
      Row 2: convergence map (plasma heatmap)
      Row 3+: per-channel mask (with border color: red=instrument, blue=tissue, gray=bg)
    """
    C, H, W = masks.shape

    # Resize everything to (H, W) — H,W = mask_size
    orig = ((img.cpu() + 2.0) / 4.0).clamp(0, 1)
    orig_r = F.interpolate(orig.unsqueeze(0), (H, W), mode='bilinear',
                           align_corners=False)[0].numpy()

    flow_rgb = flow_to_rgb(flow.cpu().numpy())

    conv_rgb = colorize_heatmap(conv_map.cpu().numpy())

    rows = [orig_r, flow_rgb, conv_rgb]

    ch_labels = []
    for c in range(C):
        m = masks[c].cpu().numpy()               # [H, W]
        m3 = np.stack([m, m, m], axis=0)         # [3, H, W]
        # tint border based on role
        if c in instrument_channels:
            m3[0] = np.where(m3[0] < 0.01, 0.8, m3[0])   # red tint on empty
        elif c == tissue_channel:
            m3[2] = np.where(m3[2] < 0.01, 0.6, m3[2])   # blue tint
        rows.append(m3)
        role = "inst" if c in instrument_channels else ("tissue?" if c == tissue_channel else "bg")
        ch_labels.append(f"ch{c}({role})")

    grid = torch.from_numpy(np.stack(rows, axis=0))  # [N, 3, H, W]
    torchvision.utils.save_image(grid, str(save_path), nrow=len(rows), padding=3)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(avg: dict, n_samples: int,
                  instrument_channels: tuple, tissue_channel: int):
    C = avg['mean_flow_mag'].shape[0]
    sep = "─" * 78

    print(f"\n{sep}")
    print(f"  Channel Role Analysis  —  averaged over {n_samples} samples")
    print(sep)
    print(f"{'Ch':>3} {'Role':>12}  {'FlowMag':>8}  {'DirVar':>8}  "
          f"{'Residual':>9}  {'Converg.':>9}  {'Area':>6}")
    print(sep)
    for c in range(C):
        role = ("Instrument" if c in instrument_channels
                else "Tissue(?)" if c == tissue_channel
                else "Background")
        print(f"{c:>3} {role:>12}"
              f"  {avg['mean_flow_mag'][c].item():>8.3f}"
              f"  {avg['flow_dir_var'][c].item():>8.3f}"
              f"  {avg['mean_residual_mag'][c].item():>9.3f}"
              f"  {avg['convergence_score'][c].item():>9.5f}"
              f"  {avg['mask_area'][c].item():>6.4f}")
    print(sep)

    instr = list(instrument_channels)
    print("\nHypothesis checks:")

    mag_i = avg['mean_flow_mag'][instr].mean().item()
    mag_t = avg['mean_flow_mag'][tissue_channel].item()
    print(f"  [FlowMag]  instrument {mag_i:.3f} vs tissue {mag_t:.3f}  "
          + ("✓ inst > tissue" if mag_i > mag_t else "✗ inst NOT > tissue"))

    var_i = avg['flow_dir_var'][instr].mean().item()
    var_t = avg['flow_dir_var'][tissue_channel].item()
    print(f"  [DirVar]   instrument {var_i:.3f} vs tissue {var_t:.3f}  "
          + ("✓ tissue more variable" if var_t > var_i else "✗ tissue NOT more variable"))

    res_i = avg['mean_residual_mag'][instr].mean().item()
    res_t = avg['mean_residual_mag'][tissue_channel].item()
    print(f"  [Residual] instrument {res_i:.3f} vs tissue {res_t:.3f}  "
          + ("✓ tissue higher residual" if res_t > res_i else "✗ tissue NOT higher residual"))

    conv_i = avg['convergence_score'][instr].mean().item()
    conv_t = avg['convergence_score'][tissue_channel].item()
    print(f"  [Converg.] instrument {conv_i:.5f} vs tissue {conv_t:.5f}  "
          + ("✓ tissue more convergent" if conv_t > conv_i else "✗ tissue NOT more convergent"))

    print()

    # Net flow direction for each channel
    print("  Net flow direction (x=right, y=down):")
    for c in range(C):
        fx = avg['mean_flow_x'][c].item()
        fy = avg['mean_flow_y'][c].item()
        role = ("inst" if c in instrument_channels
                else "tissue?" if c == tissue_channel else "bg")
        print(f"    ch{c}({role}):  dx={fx:+.3f}  dy={fy:+.3f}  "
              f"mag={np.sqrt(fx**2+fy**2):.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(model, dataset, out_dir: str,
                 n_samples: int,
                 instrument_channels: tuple,
                 tissue_channel: int,
                 n_vis: int = 15):

    out_dir = Path(out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    mask_size  = tuple(model.mask_size)           # (128, 128)
    n_channels = model.num_classes                 # C = 5

    pred_div_coeff   = model.decode_head.pred_div_coeff
    residual_scale   = model.decode_head.residual_adjustment_scale

    extractor = RCFFeatureExtractor(model)

    # Accumulate
    accum = {k: torch.zeros(n_channels) for k in [
        'mean_flow_mag', 'flow_dir_var', 'mean_residual_mag',
        'convergence_score', 'mean_flow_x', 'mean_flow_y', 'mask_area']}
    n_done = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=False)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, total=n_samples, desc="Analysing"):
            if n_done >= n_samples:
                break
            if 'gt_fw_flows' not in batch:
                continue

            batch = to_device(batch)
            imgs  = torch.stack(batch['imgs'], dim=1)  # [1, 2, 3, H, W]

            # Extract masks and residuals
            internals = extractor.extract(batch)
            soft      = internals['soft_mask']    # [1, 2, C, fH, fW]
            res_fw    = internals['res_fw']        # [1, 2*C, fH, fW]

            # Use frame-0 mask
            masks = soft[0, 0]                    # [C, fH, fW]

            # Resize masks to mask_size
            masks_r = F.interpolate(masks.unsqueeze(0), mask_size,
                                    mode='bilinear', align_corners=False)[0]  # [C, H, W]

            # Resize residual to mask_size if needed
            if res_fw.shape[-2:] != torch.Size(list(mask_size)):
                res_fw_r = F.interpolate(res_fw, mask_size, mode='bilinear',
                                         align_corners=False)
            else:
                res_fw_r = res_fw                  # [1, 2*C, H, W]

            # Get forward flow and resize to mask_size
            gt_fw = torch.stack(batch['gt_fw_flows'], dim=1)[0, 0]  # [2, Hf, Wf]
            if gt_fw.shape[-1] == 2:
                gt_fw = gt_fw.permute(2, 0, 1)
            flow_r = F.interpolate(gt_fw.unsqueeze(0), mask_size,
                                   mode='bilinear', align_corners=False)[0]  # [2, H, W]

            # Compute stats (batch_size=1 here)
            stats, conv_map = per_channel_stats(
                masks_r.unsqueeze(0),     # [1, C, H, W]
                flow_r.unsqueeze(0),      # [1, 2, H, W]
                res_fw_r,                 # [1, 2*C, H, W]
                pred_div_coeff=pred_div_coeff,
                residual_scale=residual_scale,
            )

            for k in accum:
                accum[k] += stats[k]
            n_done += 1

            # Visualize first n_vis samples
            if n_done <= n_vis:
                save_sample_vis(
                    imgs[0, 0], masks_r, flow_r, conv_map[0],
                    vis_dir / f"sample_{n_done:03d}.jpg",
                    instrument_channels, tissue_channel,
                )

    # Average
    avg = {k: v / max(n_done, 1) for k, v in accum.items()}

    print_summary(avg, n_done, instrument_channels, tissue_channel)

    # Save raw numpy stats
    np.save(str(out_dir / "channel_stats.npy"),
            {k: v.numpy() for k, v in avg.items()})
    print(f"Stats saved → {out_dir / 'channel_stats.npy'}")
    print(f"Visualizations saved → {vis_dir}/  ({min(n_done, n_vis)} frames)")

    return avg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Channel role analysis")
    p.add_argument("--config",   required=True, help="YAML config path")
    p.add_argument("--ckpt",     required=True, help="Checkpoint path")
    p.add_argument("--output",   required=True, help="Output directory")
    p.add_argument("--n_samples",     type=int, default=60,
                   help="Number of frame-pairs to analyse (default 60)")
    p.add_argument("--n_vis",         type=int, default=15,
                   help="Number of visualizations to save (default 15)")
    p.add_argument("--split",    default="ImageSets/train.txt",
                   help="Dataset split file (needs flow)")
    p.add_argument("--flow_suffix", default="_NewCT")
    p.add_argument("--instrument_channels", type=int, nargs="+", default=[0, 2, 3])
    p.add_argument("--tissue_channel",      type=int, default=1)
    args = p.parse_args()

    cfg = load_config(args.config)

    model = build_model(cfg, output_dir=args.output)
    model = load_checkpoint(model, args.ckpt)
    model.args.eval_save = False

    dataset = build_probe_dataset(cfg, split_override=args.split,
                                  flow_suffix=args.flow_suffix)

    run_analysis(
        model, dataset, args.output,
        n_samples=args.n_samples,
        n_vis=args.n_vis,
        instrument_channels=tuple(args.instrument_channels),
        tissue_channel=args.tissue_channel,
    )


if __name__ == "__main__":
    main()

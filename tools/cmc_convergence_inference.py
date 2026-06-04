#!/usr/bin/env python3
"""
CMC convergence-prior inference.

Soft-tissue post-processing: pixels whose optical flow converges toward the
grasping point are suppressed from the instrument mask.

Convergence map:
    C(x,y) = cosine_sim( flow(x,y),  direction_from_(x,y)_to_grasp_point )
    Range [-1, 1]:  +1 = pixel flow points exactly toward grasp (tissue),
                    -1 = pixel flow points exactly away (not tissue).

Refined instrument mask:
    P_refined = P_union * (1 - lambda * max(0, C))

Visualization grid (11 rows):
    1. Original image
    2-6. Channel 0-4 soft masks
    7. Union mask        (channels 0,2,3)
    8. Green overlay     (union on original)
    9. Convergence map   (hot colormap: red=converging tissue, blue=instrument)
   10. Refined mask      (union after convergence suppression)
   11. Green overlay     (refined mask on original)

Usage (with annotated grasping point):
    python tools/cmc_convergence_inference.py \
        --config  configs/instrument/rcf_cmc_grasp10_finetune_v2b.yaml \
        --ckpt    saved/cmc_grasp10_finetune_v2b_260604_120527/last.ckpt \
        --output  saved/cmc_convergence_test \
        --split   ImageSets/val.txt \
        --grasp_x 360 --grasp_y 288 \
        --lam 0.8

Usage (auto-estimate grasp point from instrument mask centroid):
    python tools/cmc_convergence_inference.py ... --auto_grasp
"""

import argparse, sys, os
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


# ─────────────────────────────────────────────────────────────────────────────
# Flow loading
# ─────────────────────────────────────────────────────────────────────────────

def load_flow(img_path: str, data_path: str):
    """Derive Flows_NewCT path from image path and load (H, W, 2) float32."""
    # img_path example: /dataset/CMC_grasp10.../JPEGImages/seq/frame.png
    flow_path = img_path.replace("JPEGImages", "Flows_NewCT")[:-4] + ".npy"
    if not os.path.exists(flow_path):
        return None
    return np.load(flow_path).astype(np.float32)   # (H, W, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence map
# ─────────────────────────────────────────────────────────────────────────────

def combined_convergence_map(flow: np.ndarray,
                             instrument_mask: np.ndarray,
                             sigma: float = 40.0) -> np.ndarray:
    """
    Combined soft-tissue prior:

        S(x,y) = C_centroid(x,y)  ×  exp(-dist(x,y) / sigma)

    where
      C_centroid = cosine_sim(flow, direction toward nearest instrument centroid)
      dist       = Euclidean distance to the nearest instrument pixel

    Properties:
      - Smooth radial direction field (centroid-based, not geometric Voronoi)
      - Distance weighting suppresses far-away noise naturally
      - Multi-instrument: each pixel directed toward the centroid of its
        nearest connected component (via distance-transform label lookup)
      - sigma (pixels) controls effective range; ~40 px focuses on the
        tissue layer immediately adjacent to the instrument

    Returns S in roughly [-1, 1]; high positive = soft tissue being pulled.
    """
    from scipy.ndimage import distance_transform_edt, label

    H, W = flow.shape[:2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    if instrument_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    # ── Distance weight ───────────────────────────────────────────────────────
    dist, nearest_idx = distance_transform_edt(
        ~instrument_mask, return_indices=True)          # dist: (H, W)
    w_dist = np.exp(-dist.astype(np.float32) / sigma)  # in [0, 1]

    # ── Per-component centroids, assigned via nearest instrument pixel ────────
    labeled, n_comp = label(instrument_mask)

    if n_comp == 1:
        iy, ix = np.where(instrument_mask)
        centroid_x = np.full((H, W), float(ix.mean()), dtype=np.float32)
        centroid_y = np.full((H, W), float(iy.mean()), dtype=np.float32)
    else:
        # For each pixel, look up which component its nearest instrument pixel
        # belongs to, then use that component's centroid as the target.
        nearest_comp = labeled[nearest_idx[0], nearest_idx[1]]  # (H, W)
        centroid_x = np.zeros((H, W), dtype=np.float32)
        centroid_y = np.zeros((H, W), dtype=np.float32)
        for c in range(1, n_comp + 1):
            iy, ix = np.where(labeled == c)
            cx, cy = float(ix.mean()), float(iy.mean())
            mask_c = (nearest_comp == c)
            centroid_x[mask_c] = cx
            centroid_y[mask_c] = cy

    # ── Direction toward nearest centroid ─────────────────────────────────────
    dx = centroid_x - xs.astype(np.float32)
    dy = centroid_y - ys.astype(np.float32)
    dir_norm = np.sqrt(dx**2 + dy**2) + 1e-8
    dx_n = dx / dir_norm
    dy_n = dy / dir_norm

    # ── Cosine similarity with flow ───────────────────────────────────────────
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]
    flow_norm = np.sqrt(fx**2 + fy**2) + 1e-8
    C_dir = (fx * dx_n + fy * dy_n) / flow_norm    # in [-1, 1]

    # ── Combine and mask out instrument interior ──────────────────────────────
    S = C_dir * w_dist
    S[instrument_mask] = 0.0
    return S.astype(np.float32)


def commotion_tissue_map(flow_corr: np.ndarray,
                         instrument_mask: np.ndarray,
                         cos_th: float = 0.5,
                         mag_th: float = 0.4,
                         abs_flow_th: float = 0.8) -> np.ndarray:
    """
    Detect soft tissue by co-motion with the instrument.

    Inspired by get_demean_affine_flow: compute mu_F = mean flow over each
    instrument connected component.  A tissue pixel "moves with" the instrument
    when its flow is both:
        (a) direction-aligned  : cos_sim(flow_corr, mu_F) > cos_th
        (b) magnitude-relevant : |flow_corr| / |mu_F| > mag_th  (moving fast enough)

    Multi-instrument: handled per connected component — each pixel is compared
    against the mu_F of its nearest instrument component.

    Post-filter: keep only soft-tissue pixels that belong to connected regions
    touching the instrument boundary (removes distant coincidental co-movers).

    Returns binary mask (float32, 0/1).
    """
    from scipy.ndimage import distance_transform_edt, label, binary_dilation

    H, W = flow_corr.shape[:2]

    if instrument_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    # ── Per-component mean flow (mu_F) ────────────────────────────────────────
    labeled, n_comp = label(instrument_mask)
    _, nearest_idx  = distance_transform_edt(
        ~instrument_mask, return_indices=True)
    nearest_comp = labeled[nearest_idx[0], nearest_idx[1]]   # (H, W)

    mu_fx = np.zeros((H, W), dtype=np.float32)
    mu_fy = np.zeros((H, W), dtype=np.float32)
    comp_mu_mags = []
    for c in range(1, n_comp + 1):
        comp = (labeled == c)
        mfx  = float(flow_corr[:, :, 0][comp].mean())
        mfy  = float(flow_corr[:, :, 1][comp].mean())
        comp_mu_mags.append(np.sqrt(mfx**2 + mfy**2))
        region = (nearest_comp == c)
        mu_fx[region] = mfx
        mu_fy[region] = mfy

    # ── Skip if instrument is barely moving (camera-relative) ────────────────
    # When mu_F ≈ 0 the co-motion signal is meaningless (instrument at rest).
    # abs_flow_th is reused as the minimum instrument speed to attempt detection.
    max_comp_speed = float(np.max(comp_mu_mags))
    if max_comp_speed < abs_flow_th:
        return np.zeros((H, W), dtype=np.float32)

    # ── Co-motion score ───────────────────────────────────────────────────────
    fx = flow_corr[:, :, 0]
    fy = flow_corr[:, :, 1]
    flow_mag = np.sqrt(fx**2 + fy**2) + 1e-8
    mu_mag   = np.sqrt(mu_fx**2 + mu_fy**2) + 1e-8

    cos_sim  = (fx * mu_fx + fy * mu_fy) / (flow_mag * mu_mag)  # [-1, 1]
    rel_mag  = flow_mag / mu_mag                                  # ratio

    # ── Threshold ─────────────────────────────────────────────────────────────
    candidate = (
        (cos_sim  > cos_th)   &    # direction aligns with instrument motion
        (rel_mag  > mag_th)   &    # pixel moves ≥ mag_th × instrument speed
        (~instrument_mask)
    )

    # ── Keep only connected regions that touch the instrument boundary ─────────
    # Dilate instrument by 1 px to find "seed" contact pixels
    seed_zone = binary_dilation(instrument_mask, iterations=2) & ~instrument_mask
    seeds     = candidate & seed_zone

    # Flood-fill: label candidates, keep components that contain a seed
    cand_labeled, n_cand = label(candidate)
    seed_labels = set(cand_labeled[seeds].tolist()) - {0}

    result = np.zeros((H, W), dtype=np.float32)
    for lbl in seed_labels:
        result[cand_labeled == lbl] = 1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Find flow convergence point via weighted least-squares line intersection
# ─────────────────────────────────────────────────────────────────────────────

def subtract_affine_motion(flow: np.ndarray,
                           tissue_mask: np.ndarray) -> np.ndarray:
    """
    Estimate global camera motion as an affine flow field fitted to background
    (tissue far from instrument), then subtract it per-pixel.

    The affine model is:
        F_pred(x, y) = A @ [[x - cx], [y - cy]] + t
    where A is a 2x2 matrix and t is a 2-vector (translation at centroid).
    This removes translation + rotation + zoom + shear — all components of
    laparoscope rigid-body motion — unlike the old median approach which only
    removed pure translation.
    """
    from scipy.ndimage import binary_erosion
    H, W = flow.shape[:2]

    # Use eroded tissue mask as "pure background" (far from instrument)
    bg_mask = binary_erosion(tissue_mask > 0.5, iterations=10)
    if bg_mask.sum() < 30:
        bg_mask = tissue_mask > 0.5      # fallback: all tissue pixels

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    x_bg = xs[bg_mask].astype(np.float64)
    y_bg = ys[bg_mask].astype(np.float64)
    fx_bg = flow[:, :, 0][bg_mask].astype(np.float64)
    fy_bg = flow[:, :, 1][bg_mask].astype(np.float64)

    # Centre coordinates to improve numerical conditioning
    cx, cy = x_bg.mean(), y_bg.mean()
    xc = x_bg - cx
    yc = y_bg - cy

    # Design matrix: [x-cx, y-cy, 1]  →  fits A (2×2) + translation (2,)
    D = np.column_stack([xc, yc, np.ones_like(xc)])   # (N, 3)
    params_fx, _, _, _ = np.linalg.lstsq(D, fx_bg, rcond=None)
    params_fy, _, _, _ = np.linalg.lstsq(D, fy_bg, rcond=None)

    # Predict over the whole image
    xc_full = (xs.ravel() - cx).astype(np.float64)
    yc_full = (ys.ravel() - cy).astype(np.float64)
    D_full  = np.column_stack([xc_full, yc_full, np.ones_like(xc_full)])

    pred_fx = (D_full @ params_fx).reshape(H, W).astype(np.float32)
    pred_fy = (D_full @ params_fy).reshape(H, W).astype(np.float32)

    corrected = flow.copy()
    corrected[:, :, 0] -= pred_fx
    corrected[:, :, 1] -= pred_fy
    return corrected




# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_cross(img: torch.Tensor, cx: int, cy: int,
               size: int = 12, thickness: int = 2,
               color=(1.0, 1.0, 0.0)) -> torch.Tensor:
    """Draw a yellow cross at (cx, cy) on img [1, 3, H, W] in-place."""
    img = img.clone()
    H, W = img.shape[-2], img.shape[-1]
    cx = int(np.clip(cx, 0, W - 1))
    cy = int(np.clip(cy, 0, H - 1))
    # Horizontal bar
    x0, x1 = max(0, cx - size), min(W, cx + size + 1)
    y0, y1 = max(0, cy - thickness), min(H, cy + thickness + 1)
    img[0, 0, y0:y1, x0:x1] = color[0]
    img[0, 1, y0:y1, x0:x1] = color[1]
    img[0, 2, y0:y1, x0:x1] = color[2]
    # Vertical bar
    x0, x1 = max(0, cx - thickness), min(W, cx + thickness + 1)
    y0, y1 = max(0, cy - size), min(H, cy + size + 1)
    img[0, 0, y0:y1, x0:x1] = color[0]
    img[0, 1, y0:y1, x0:x1] = color[1]
    img[0, 2, y0:y1, x0:x1] = color[2]
    return img


def make_green_overlay(orig_r: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """orig_r [1,3,H,W] in [0,1],  mask [H,W] binary → overlay [1,3,H,W]."""
    ov = orig_r.clone()
    m  = mask.bool()
    ov[0, 0][m] = ov[0, 0][m] * 0.3
    ov[0, 1][m] = (ov[0, 1][m] * 0.5 + 180 / 255.).clamp(0, 1)
    ov[0, 2][m] = ov[0, 2][m] * 0.3
    return ov


def convergence_colormap(C: np.ndarray) -> torch.Tensor:
    """
    White-centre diverging colormap with depth gradient:
        strong convergence (+) → dark red    (white → orange → dark red)
        neutral (~0)           → white
        divergence      (-)    → dark blue   (white → light blue → dark blue)

    Avoids tanh saturation so mid-range differences remain visible.
    """
    # Per-frame normalisation using 90th-percentile of abs values
    scale = max(float(np.percentile(np.abs(C), 90)), 1e-3)
    C_norm = np.clip(C / scale, -1.0, 1.0).astype(np.float32)

    t   = torch.from_numpy(C_norm)
    pos = t.clamp(0, 1)     # convergence  [0, 1]
    neg = (-t).clamp(0, 1)  # divergence   [0, 1]

    # Convergence: white → light red → dark red
    #   r: stays near 1, drops slightly at max
    #   g, b: drop together → moves from white toward (0.4, 0, 0) dark red
    r = 1.0 - 0.6 * pos
    g = 1.0 - pos
    b = 1.0 - pos

    # Divergence: overwrite with white → light blue → dark blue
    #   b: stays near 1
    #   r, g: drop together
    r = torch.where(neg > 0, 1.0 - neg,       r)
    g = torch.where(neg > 0, 1.0 - neg,       g)
    b = torch.where(neg > 0, 1.0 - 0.6 * neg, b)

    rgb = torch.stack([r, g, b], dim=0)[None]
    return rgb.clamp(0, 1)


def resize_to(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(t, (h, w), mode='bilinear', align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, dataset, out_dir: str,
                  union_channels=(0, 2, 3),
                  lam: float = 0.8,
                  sigma: float = 40.0,
                  cos_th: float = 0.5,
                  mag_th: float = 0.4,
                  abs_flow_th: float = 0.8,
                  eval_pos_th: float = 0.35,
                  workers: int = 0):

    out_dir = Path(out_dir)
    save_dir = out_dir / "saved_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_dir_eval = str(save_dir)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=False)

    print(f"Inference on {len(dataset)} frames → {save_dir}")
    print("Convergence mode: boundary distance transform (per-pixel nearest instrument)")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Convergence inference"):
            batch   = to_device(batch)
            imgs    = torch.stack(batch["imgs"], dim=1)   # [1,1,3,H,W]
            img_path = batch["paths"][0][0]               # raw image path

            # ── 1. Model forward ─────────────────────────────────────────────
            pred_masks = model.forward_eval(
                imgs, batch["seq_ids"], batch["seq_names"], batch["paths"])
            # pred_masks: [1, C, fh, fw]

            p = model._pending_eval_viz
            if p is None:
                continue

            row_h, row_w = p["row_h"], p["row_w"]
            base   = p["tosave"]          # [1, 3, 6*row_h, row_w]
            device = base.device

            # ── 2. Soft masks at row resolution ──────────────────────────────
            soft   = F.softmax(pred_masks, dim=1)
            soft_r = resize_to(soft, row_h, row_w)   # [1,C,row_h,row_w]

            # Union mask (instrument channels)
            union_prob = soft_r[0, union_channels[0]].clone()
            for ch in union_channels[1:]:
                union_prob = torch.max(union_prob, soft_r[0, ch])
            union_bin = (union_prob > eval_pos_th).float()   # [row_h,row_w]

            # ── 3. Original image at row resolution ──────────────────────────
            orig   = imgs[0, 0]                       # [3,H,W]
            orig_r = resize_to(orig[None], row_h, row_w)
            orig_r = ((orig_r + 2.0) / 4.0).clamp(0, 1)

            # ── 4. Load optical flow ─────────────────────────────────────────
            flow_np = load_flow(img_path, "")         # (H_orig, W_orig, 2) or None

            if flow_np is not None:
                H_f, W_f = flow_np.shape[:2]

                # ── 5. Instrument mask at flow resolution ────────────────────
                union_orig = F.interpolate(
                    union_bin[None, None].float(), (H_f, W_f),
                    mode='nearest')[0, 0].cpu().numpy()
                instrument_mask = (union_orig > 0.5)
                tissue_mask     = ~instrument_mask

                # Subtract affine camera motion (uses far tissue as background)
                flow_corr = subtract_affine_motion(flow_np, tissue_mask)

                # ── 6. Combined convergence map ───────────────────────────────
                # S = cos_sim(flow, →centroid) × exp(-dist / sigma)
                # Smooth direction + distance decay; multi-instrument aware.
                C_orig = combined_convergence_map(
                    flow_corr, instrument_mask, sigma=sigma)
                C_t    = torch.from_numpy(C_orig)[None, None].float()
                C_r    = resize_to(C_t, row_h, row_w)[0, 0]
                # Protect eroded instrument core from upsampling bleed
                from scipy.ndimage import binary_erosion as _be
                inst_core = _be(union_bin.cpu().numpy().astype(bool), iterations=3)
                C_r[torch.from_numpy(inst_core)] = 0.0

                # ── 7. Refined mask ───────────────────────────────────────────
                suppression  = (lam * C_r.clamp(0, 1)).to(device)
                refined_prob = union_prob * (1.0 - suppression)
                refined_bin  = (refined_prob > eval_pos_th).float()

                # ── 8a. Co-motion soft-tissue mask ────────────────────────────
                # Use RAW flow (not affine-corrected) for co-motion detection.
                # Affine correction removes the bulk motion shared by instrument
                # and tissue, destroying the very co-motion signal we need.
                # Raw flow: instrument moves → grasped tissue moves with it →
                # both have similar flow vectors vs stationary background.
                soft_np = commotion_tissue_map(
                    flow_corr, instrument_mask,
                    cos_th=cos_th, mag_th=mag_th,
                    abs_flow_th=abs_flow_th)
                soft_t  = torch.from_numpy(soft_np)[None, None].float()
                soft_tissue_bin = resize_to(soft_t, row_h, row_w)[0, 0].to(device)

                # Instrument centroid cross (visualization reference only)
                iy, ix = np.where(instrument_mask)
                if len(ix) > 0:
                    cx_r = int(round(ix.mean() * row_w / W_f))
                    cy_r = int(round(iy.mean() * row_h / H_f))
                else:
                    cx_r, cy_r = row_w // 2, row_h // 2

                # Convergence colormap row
                conv_row = resize_to(
                    convergence_colormap(C_orig), row_h, row_w).to(device)
                conv_row = draw_cross(conv_row.cpu(), cx_r, cy_r).to(device)
            else:
                cx_r, cy_r      = row_w // 2, row_h // 2
                C_r             = torch.zeros(row_h, row_w)
                refined_bin     = union_bin
                soft_tissue_bin = torch.zeros(row_h, row_w)
                conv_row        = torch.ones(1, 3, row_h, row_w).to(device) * 0.5

            # ── 8. Build visualization rows ────────────────────────────────
            union_row = union_bin[None, None].repeat(1, 3, 1, 1).to(device)
            overlay_u = make_green_overlay(orig_r.cpu(), union_bin.cpu()).to(device)
            overlay_u = draw_cross(overlay_u.cpu(), cx_r, cy_r).to(device)
            overlay_r = make_green_overlay(orig_r.cpu(), refined_bin.cpu())
            overlay_r = draw_cross(overlay_r, cx_r, cy_r).to(device)

            # Soft-tissue overlay: fluorescent blue (0, 220, 255)
            soft_overlay = orig_r.clone().cpu()
            st_m = soft_tissue_bin.bool().cpu()
            soft_overlay[0, 0][st_m] = soft_overlay[0, 0][st_m] * 0.2
            soft_overlay[0, 1][st_m] = (soft_overlay[0, 1][st_m] * 0.3 + 220/255.).clamp(0, 1)
            soft_overlay[0, 2][st_m] = (soft_overlay[0, 2][st_m] * 0.3 + 255/255.).clamp(0, 1)
            soft_overlay = soft_overlay.to(device)

            # Grid: original(1) + channels(5) + union(1) + union_overlay(1)
            #     + conv_map(1) + refined_overlay(1) + soft_tissue_overlay(1) = 11
            tosave = torch.cat(
                [base, union_row, overlay_u, conv_row, overlay_r, soft_overlay],
                dim=2)

            model._pending_eval_viz = None
            model.save_eval_visualizations(
                tosave, p["paths"], p["seq_ids"], p["seq_names"],
                train_iter=model.train_iter)

    print(f"Done → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",        required=True)
    ap.add_argument("--ckpt",          required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--split",         default="ImageSets/val.txt")
    ap.add_argument("--union_channels",type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--lam",           type=float, default=0.8,
                    help="Convergence suppression strength (0=none, 1=full).")
    ap.add_argument("--sigma",         type=float, default=40.0,
                    help="Distance decay (pixels) for instrument-mask refinement.")
    ap.add_argument("--cos_th",        type=float, default=0.5,
                    help="Cosine similarity threshold for co-motion tissue detection.")
    ap.add_argument("--mag_th",        type=float, default=0.4,
                    help="Relative flow magnitude threshold (|flow|/|mu_F|) for co-motion.")
    ap.add_argument("--abs_flow_th",   type=float, default=0.8,
                    help="Absolute flow magnitude floor (pixels) to guard against tiny mu_F.")
    ap.add_argument("--pos_th",        type=float, default=0.35)
    ap.add_argument("--workers",       type=int,   default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["model_kwargs"]["allow_mask_resize"] = True
    model   = build_model(cfg, output_dir=args.output)
    model   = load_checkpoint(model, args.ckpt)
    dataset = build_dataset(cfg, split_override=args.split)

    run_inference(model, dataset, args.output,
                  union_channels=tuple(args.union_channels),
                  lam=args.lam,
                  sigma=args.sigma,
                  cos_th=args.cos_th,
                  mag_th=args.mag_th,
                  abs_flow_th=args.abs_flow_th,
                  eval_pos_th=args.pos_th,
                  workers=args.workers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
grasp0_flow_analysis.py

For single-annotation grasp0 sequences, compute and visualise:
  1. Original image + markers
  2. Flow magnitude
  3. Flow direction (HSV)
  4. Gradient magnitude (motion boundary)
  5. Divergence  (du/dx + dv/dy)  — positive=expand, negative=shrink
  6. Curl/Rotation  (dv/dx - du/dy)
  7. Shear  (du/dy + dv/dx)
  8. Anisotropic stretch  (du/dx - dv/dy)
  9. Radial cosine similarity toward G point

Output in analysis/flow_annotation_<ts>/:
  - <seq>.png   3×3 grid (each panel is a panel for one metric)
  - summary.csv
  - flow_stats.txt
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

SAMPLE_R  = 15
RADIAL_R1 = 10
RADIAL_R2 = 60
VIS_W     = 720   # rescale wide images to this width for the grid

LIME = (57, 255, 20)
CYAN = (0, 200, 255)


# ── Annotation loader ──────────────────────────────────────────────────────────

def load_anns(ann_dir: Path, seqs: list):
    result = []
    for seq in seqs:
        f = ann_dir / f'{seq}.json'
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        anns = []
        for ad in d.get('annotations', []):
            for k, pts in ad.items():
                if len(pts) >= 2:
                    anns.append((k, pts[0], pts[1]))
        if len(anns) == 1:
            result.append((seq, *anns[0]))
    return result


# ── Flow field decomposition ───────────────────────────────────────────────────

def jacobian(flow: np.ndarray):
    """Return (du_dx, du_dy, dv_dx, dv_dy)."""
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return du_dx, du_dy, dv_dx, dv_dy


def _lap(arr: np.ndarray) -> np.ndarray:
    """Laplacian ∂²/∂x² + ∂²/∂y² of a 2-D scalar field."""
    return (np.gradient(np.gradient(arr.astype(np.float32), axis=1), axis=1) +
            np.gradient(np.gradient(arr.astype(np.float32), axis=0), axis=0)).astype(np.float32)


def local_flow_variance(flow: np.ndarray, k: int = 7) -> np.ndarray:
    """Local flow variance magnitude over a k×k neighbourhood.

    Returns sqrt(Var[u] + Var[v]) where variance is computed per-pixel over
    the local window via uniform_filter.  High values occur at boundaries between
    regions with different motion patterns — naturally smoother and less noisy
    than gradient/Laplacian approaches.
    """
    from scipy.ndimage import uniform_filter
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    mean_u  = uniform_filter(u,    size=k)
    mean_u2 = uniform_filter(u**2, size=k)
    mean_v  = uniform_filter(v,    size=k)
    mean_v2 = uniform_filter(v**2, size=k)
    var_u = np.clip(mean_u2 - mean_u**2, 0, None)
    var_v = np.clip(mean_v2 - mean_v**2, 0, None)
    return np.sqrt(var_u + var_v).astype(np.float32)


def gauss_grad_mag(flow: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gradient magnitude of flow magnitude after Gaussian smoothing.

    Pre-smoothing suppresses isolated noise points while preserving real
    motion boundaries — a lightweight comparison baseline against local variance.
    """
    from scipy.ndimage import gaussian_filter
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    mag = np.sqrt(u**2 + v**2)
    mag_s = gaussian_filter(mag, sigma=sigma)
    gx = np.gradient(mag_s, axis=1)
    gy = np.gradient(mag_s, axis=0)
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def flow_kmeans(flow: np.ndarray, k: int):
    """Cluster per-pixel (u, v) flow vectors into k motion groups.

    Returns:
      labels  (H, W) int32  — cluster assignment 0..k-1
      label_rgb (H, W, 3) uint8 — distinct colour per cluster
      boundary  (H, W) uint8  — 255 at pixels adjacent to a different cluster
    """
    from scipy.cluster.vq import kmeans2, whiten
    H, W = flow.shape[:2]
    pts = flow.reshape(-1, 2).astype(np.float32)
    pts_w = whiten(pts + 1e-6)          # normalise each feature to unit std
    np.random.seed(0)
    _, labels = kmeans2(pts_w, k, iter=20, minit='points', missing='raise', seed=0)
    labels = labels.reshape(H, W).astype(np.int32)

    # distinct colours (tab10 palette)
    palette = np.array([
        [31,119,180], [255,127,14], [44,160,44], [214,39,40],
        [148,103,189],[140,86,75], [227,119,194],[127,127,127],
        [188,189,34], [23,190,207],
    ], dtype=np.uint8)
    label_rgb = palette[labels % len(palette)]

    # boundary: pixel differs from any 4-neighbour
    diff = np.zeros((H, W), dtype=bool)
    diff[:-1, :] |= (labels[:-1, :] != labels[1:,  :])
    diff[1:,  :] |= (labels[1:,  :] != labels[:-1, :])
    diff[:, :-1] |= (labels[:, :-1] != labels[:, 1: ])
    diff[:, 1: ] |= (labels[:, 1: ] != labels[:, :-1])
    boundary = (diff * 255).astype(np.uint8)

    return labels, label_rgb, boundary


def _boundary_overlay(img_rgb: np.ndarray, boundary: np.ndarray,
                      color=(255, 255, 0)) -> np.ndarray:
    """Overlay boundary pixels (white by default) on an RGB image."""
    out = img_rgb.copy()
    mask = boundary > 0
    out[mask] = color
    return out


def filter_variance_boundary(var_map: np.ndarray,
                              thresh_pct: float = 80,
                              min_area: int = 150) -> np.ndarray:
    """Remove isolated noise from a variance/gradient map via connected-component filtering.

    Steps:
      1. Threshold at `thresh_pct`-th percentile → binary mask
      2. Label connected components (8-connectivity)
      3. Zero out components whose pixel count < min_area  (isolated dots/specks)
      4. Return float32 map: original var_map values where kept, 0 elsewhere

    The result retains thick boundary bands (not thinned) but removes scattered
    single-pixel noise that pollutes derivative-based maps.
    """
    from scipy.ndimage import label as nd_label
    thresh = np.percentile(var_map, thresh_pct)
    binary = (var_map >= thresh).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)          # 8-connectivity
    labeled, n = nd_label(binary, structure=struct)
    keep = np.zeros_like(binary, dtype=bool)
    for comp_id in range(1, n + 1):
        if (labeled == comp_id).sum() >= min_area:
            keep |= (labeled == comp_id)
    out = np.where(keep, var_map, 0.0).astype(np.float32)
    return out


def flow_maps(flow: np.ndarray):
    """Compute all derived scalar maps (1st and 2nd order) from the flow field."""
    du_dx, du_dy, dv_dx, dv_dy = jacobian(flow)
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    # ── 0th order ─────────────────────────────────────────────────────────────
    mag = np.sqrt(u**2 + v**2)

    # ── 1st order ─────────────────────────────────────────────────────────────
    grad_mag = np.sqrt(np.gradient(mag, axis=1)**2 +
                       np.gradient(mag, axis=0)**2).astype(np.float32)
    div     = (du_dx + dv_dy).astype(np.float32)
    curl    = (dv_dx - du_dy).astype(np.float32)
    shear   = (du_dy + dv_dx).astype(np.float32)
    stretch = (du_dx - dv_dy).astype(np.float32)
    strain  = np.sqrt(du_dx**2 + dv_dy**2 + 0.5*(du_dy+dv_dx)**2).astype(np.float32)

    # ── 2nd order ─────────────────────────────────────────────────────────────
    lap_div = _lap(div)          # ∇²(div)  — negative = convergence centre
    lap_mag = _lap(mag)          # ∇²(mag)  — negative = motion centre
    # vector Laplacian magnitude: sqrt((∇²u)² + (∇²v)²)
    lap_vec = np.sqrt(_lap(u)**2 + _lap(v)**2)
    # Hessian determinant of mag: ∂²mag/∂x² · ∂²mag/∂y² − (∂²mag/∂x∂y)²
    d2x  = np.gradient(np.gradient(mag, axis=1), axis=1).astype(np.float32)
    d2y  = np.gradient(np.gradient(mag, axis=0), axis=0).astype(np.float32)
    d2xy = np.gradient(np.gradient(mag, axis=1), axis=0).astype(np.float32)
    hess_det = (d2x * d2y - d2xy**2).astype(np.float32)

    return mag, grad_mag, div, curl, shear, stretch, strain, \
           lap_div, lap_mag, lap_vec, hess_det


# ── Point-based metrics ────────────────────────────────────────────────────────

def local_mean(arr: np.ndarray, cx: int, cy: int, r: int) -> float:
    H, W = arr.shape
    y0, y1 = max(0, cy-r), min(H, cy+r+1)
    x0, x1 = max(0, cx-r), min(W, cx+r+1)
    return float(arr[y0:y1, x0:x1].mean())


def local_flow_stats(flow: np.ndarray, cx: int, cy: int, r: int):
    H, W = flow.shape[:2]
    y0, y1 = max(0, cy-r), min(H, cy+r+1)
    x0, x1 = max(0, cx-r), min(W, cx+r+1)
    p = flow[y0:y1, x0:x1]
    return float(p[...,0].mean()), float(p[...,1].mean()), \
           float(np.sqrt(p[...,0]**2 + p[...,1]**2).mean())


def radial_cosine(flow: np.ndarray, cx: int, cy: int,
                  r1: int = RADIAL_R1, r2: int = RADIAL_R2) -> float:
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    dx = cx - xs; dy = cy - ys
    dist = np.sqrt(dx**2 + dy**2); dist[dist==0] = 1
    dx_n = dx/dist; dy_n = dy/dist
    fu = flow[...,0]; fv = flow[...,1]
    fmag = np.sqrt(fu**2 + fv**2); fmag[fmag==0] = 1
    cos = (fu/fmag)*dx_n + (fv/fmag)*dy_n
    mask = (dist >= r1) & (dist <= r2)
    return float(cos[mask].mean()) if mask.any() else 0.0


def radial_cos_map(flow: np.ndarray, cx: int, cy: int) -> np.ndarray:
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    dx = cx - xs; dy = cy - ys
    dist = np.sqrt(dx**2 + dy**2); dist[dist==0] = 1
    dx_n = dx/dist; dy_n = dy/dist
    fu = flow[...,0]; fv = flow[...,1]
    fmag = np.sqrt(fu**2 + fv**2); fmag[fmag==0] = 1
    return ((fu/fmag)*dx_n + (fv/fmag)*dy_n).astype(np.float32)


# ── Unsupervised soft-tissue candidate ────────────────────────────────────────

def soft_candidate_score(div_diff: np.ndarray, strain: np.ndarray) -> np.ndarray:
    """Unsupervised soft-tissue candidate score (no annotation needed).

    Logic:
      div_diff = div_fw - div_bw  → large |value| where real deformation occurs
                                    (camera motion cancels: div_fw ≈ div_bw → diff ≈ 0)
      strain                      → large where tissue deforms

    score = |div_diff| * strain
      - non-negative everywhere
      - high value = strong fw/bw asymmetry AND large deformation → likely soft tissue
      - captures both convergence (div_diff < 0) and expansion (div_diff > 0)
    """
    change = np.abs(div_diff).astype(np.float32)
    return (change * strain).astype(np.float32)


# ── Colourmaps ─────────────────────────────────────────────────────────────────

def cm_magnitude(arr: np.ndarray) -> np.ndarray:
    """Blue→green→yellow for non-negative magnitude."""
    vmax = np.percentile(arr, 98) + 1e-6
    t = np.clip(arr / vmax, 0, 1)
    r = np.clip(t*2 - 1, 0, 1)
    g = np.clip(t*2,     0, 1)
    b = np.clip(1 - t*2, 0, 1)
    return (np.stack([r,g,b], axis=-1) * 255).astype(np.uint8)


def cm_diverging(arr: np.ndarray, pct: float = 98) -> np.ndarray:
    """Red (+) → white (0) → blue (−)."""
    vmax = np.percentile(np.abs(arr), pct) + 1e-6
    t = np.clip(arr / vmax, -1, 1)
    pos = np.clip(t,  0, 1)
    neg = np.clip(-t, 0, 1)
    mid = 1 - pos - neg
    r = np.clip(pos + mid, 0, 1)
    g = np.clip(mid,       0, 1)
    b = np.clip(neg + mid, 0, 1)
    return (np.stack([r,g,b], axis=-1) * 255).astype(np.uint8)


def cm_flow_direction(flow: np.ndarray) -> np.ndarray:
    """HSV: hue=direction, saturation=normalised magnitude, value=1."""
    u, v = flow[...,0], flow[...,1]
    angle = np.arctan2(v, u)
    h = ((angle + np.pi) / (2*np.pi)).astype(np.float32)
    mag = np.sqrt(u**2 + v**2).astype(np.float32)
    vmax = np.percentile(mag, 98) + 1e-6
    s = np.clip(mag / vmax, 0, 1).astype(np.float32)
    ones = np.ones_like(h)
    hsv = np.stack([h, s, ones], axis=-1)
    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


# ── Per-panel label ────────────────────────────────────────────────────────────

def _label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    tw = len(text) * 11
    draw.rectangle([2, 2, tw + 4, 24], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return img


def _mark(arr_rgb: np.ndarray, dx=None, dy=None, gx=None, gy=None,
          r=10, cross=16, w=2) -> Image.Image:
    im = Image.fromarray(arr_rgb, 'RGB')
    if dx is None:
        return im
    drw = ImageDraw.Draw(im)
    for cx, cy, col in [(dx, dy, LIME), (gx, gy, CYAN)]:
        drw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col, width=w)
        drw.line([cx-cross, cy, cx+cross, cy], fill=col, width=w)
        drw.line([cx, cy-cross, cx, cy+cross], fill=col, width=w)
    return im


# ── Main visualisation: 3×3 grid ───────────────────────────────────────────────

def make_vis(img_rgb: Image.Image, fw: np.ndarray, bw: np.ndarray,
             out_path: Path, d_xy: list = None, g_xy: list = None):
    """d_xy / g_xy: normalised [x,y] coordinates; pass None to skip markers and G-cos."""
    # optional downscale for large frames
    W0, H0 = img_rgb.size
    if W0 > VIS_W:
        scale = VIS_W / W0
        W1, H1 = VIS_W, int(H0 * scale)
        img_rgb = img_rgb.resize((W1, H1), Image.BILINEAR)
        def _rs(f): return np.stack([
            np.array(Image.fromarray(f[...,c]).resize((W1,H1), Image.BILINEAR))
            for c in range(2)], axis=-1)
        fw = _rs(fw); bw = _rs(bw)
        W, H = W1, H1
    else:
        W, H = W0, H0

    have_ann = d_xy is not None and g_xy is not None
    dx = int(d_xy[0]*W) if have_ann else None
    dy = int(d_xy[1]*H) if have_ann else None
    gx = int(g_xy[0]*W) if have_ann else None
    gy = int(g_xy[1]*H) if have_ann else None

    # ── forward maps ──────────────────────────────────────────────────────────
    mag, grad_mag, div, curl, shear, stretch, strain, \
        lap_div, lap_mag, lap_vec, hess_det = flow_maps(fw)

    # ── bidirectional maps ────────────────────────────────────────────────────
    consist = np.sqrt((fw[...,0]+bw[...,0])**2 + (fw[...,1]+bw[...,1])**2)
    _, _, div_bw, _, _, _, _, _, _, _, _ = flow_maps(bw)
    div_diff = div - div_bw

    # ── panel 14: annotation-dependent or unsupervised fallback ──────────────
    if have_ann:
        cos_map  = radial_cos_map(fw, gx, gy)
        p14_arr  = cm_diverging(cos_map)
        p14_name = 'G-cos (red=toward G)'
    else:
        cand     = soft_candidate_score(div_diff, strain)
        p14_arr  = cm_magnitude(cand)
        p14_name = 'Soft candidate (-div_diff * strain)'

    def mk(arr): return _mark(arr, dx, dy, gx, gy)

    panels = [
        # ── Row 1: raw signal ───────────────────────────────────────────────
        _label(mk(np.array(img_rgb)),        'Original'),
        _label(mk(cm_magnitude(mag)),         'Flow magnitude'),
        _label(mk(cm_flow_direction(fw)),     'Flow direction (HSV)'),
        # ── Row 2: 1st-order ────────────────────────────────────────────────
        _label(mk(cm_magnitude(grad_mag)),    'Gradient (motion boundary)'),
        _label(mk(cm_diverging(div)),         'Div fw (red=expand)'),
        _label(mk(cm_diverging(curl)),        'Curl / Rotation'),
        # ── Row 3: 1st-order (cont.) + strain ──────────────────────────────
        _label(mk(cm_diverging(shear)),       'Shear'),
        _label(mk(cm_diverging(stretch)),     'Anisotropic stretch'),
        _label(mk(cm_magnitude(strain)),      'Strain ||e||'),
        # ── Row 4: 2nd-order ────────────────────────────────────────────────
        _label(mk(cm_diverging(lap_div)),     'Lap(div) conv.ctr'),
        _label(mk(cm_diverging(lap_mag)),     'Lap(mag) motion ctr'),
        _label(mk(cm_magnitude(lap_vec)),     'Lap(u,v) vec mag'),
        # ── Row 5: Hessian + cosine/candidate ───────────────────────────────
        _label(mk(cm_diverging(hess_det)),    'Hessian det blob'),
        _label(mk(p14_arr),                   p14_name),
        # ── Row 6: bidirectional ─────────────────────────────────────────────
        _label(mk(cm_magnitude(consist)),     'Consistency err (low=reliable)'),
        _label(mk(cm_diverging(div_bw)),      'Div bw (red=expand)'),
        _label(mk(cm_diverging(div_diff)),    'Div fw-bw (blue=converge)'),
    ]

    # ── Row 7: local flow variance (motion-boundary, smooth) ─────────────────
    lvar7  = local_flow_variance(fw, k=7)
    lvar15 = local_flow_variance(fw, k=15)
    ggrad  = gauss_grad_mag(fw, sigma=2.0)
    panels += [
        _label(mk(cm_magnitude(lvar7)),  'LocalVar k=7  (boundary smooth)'),
        _label(mk(cm_magnitude(lvar15)), 'LocalVar k=15 (large-scale)'),
        _label(mk(cm_magnitude(ggrad)),  'GaussGrad σ=2 (baseline)'),
    ]

    # ── Row 8: connected-component filtered (noise removal) ───────────────────
    fvar7_80  = filter_variance_boundary(lvar7,  thresh_pct=80, min_area=150)
    fvar7_90  = filter_variance_boundary(lvar7,  thresh_pct=90, min_area=150)
    fvar15_80 = filter_variance_boundary(lvar15, thresh_pct=80, min_area=150)
    panels += [
        _label(mk(cm_magnitude(fvar7_80)),  'LocalVar k=7  CC-filter p80'),
        _label(mk(cm_magnitude(fvar7_90)),  'LocalVar k=7  CC-filter p90'),
        _label(mk(cm_magnitude(fvar15_80)), 'LocalVar k=15 CC-filter p80'),
    ]

    # ── Row 9+10: K-means flow clustering k=2,3,4 ────────────────────────────
    img_arr = np.array(img_rgb)   # already resized above
    km_rows_clusters  = []
    km_rows_overlay   = []
    for k in (2, 3, 4):
        _, lrgb, bnd = flow_kmeans(fw, k=k)
        km_rows_clusters.append(
            _label(mk(lrgb),                            f'KMeans k={k} clusters'))
        km_rows_overlay.append(
            _label(mk(_boundary_overlay(img_arr, bnd)), f'KMeans k={k} boundary on img'))
    panels += km_rows_clusters + km_rows_overlay

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new('RGB', (W * cols, H * rows))
    for idx, panel in enumerate(panels):
        c, r = idx % cols, idx // cols
        grid.paste(panel, (c*W, r*H))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out_path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',     default='/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
    p.add_argument('--split',    default='ImageSets/trainval_single.txt')
    p.add_argument('--ann_dir',  default='grasping_points/grasp_dissect_annotations')
    p.add_argument('--out_base', default='analysis/flow_annotation')
    p.add_argument('--no_vis',   action='store_true')
    p.add_argument('--use_ann',  action='store_true',
                   help='Load grasp annotations and show D/G markers + G-cos panel. '
                        'Without this flag, processes all sequences unsupervised.')
    p.add_argument('--max_seqs', type=int, default=None,
                   help='Randomly sample this many sequences (reproducible with --seed).')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    root     = Path(args.data)
    ann_dir  = root / args.ann_dir
    flow_dir = root / 'Flows_NewCT'
    bw_dir   = root / 'BackwardFlows_NewCT'
    ts       = datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir  = root / f'{args.out_base}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output → {out_dir}')

    seqs = [Path(l.split()[0]).name
            for l in (root/args.split).read_text().splitlines() if l.strip()]

    # ── Optional random subsample ──────────────────────────────────────────────
    if args.max_seqs and args.max_seqs < len(seqs):
        import random
        random.seed(args.seed)
        seqs = random.sample(seqs, args.max_seqs)
        print(f'Sampled {len(seqs)} sequences (seed={args.seed})')

    # ── Build (seq, d_xy, g_xy) list depending on mode ────────────────────────
    if args.use_ann:
        ann_entries = load_anns(ann_dir, seqs)
        # entries: (seq, frame_key, d_xy, g_xy) — only single-annotation seqs
        work_items = [(seq, d_xy, g_xy) for seq, _, d_xy, g_xy in ann_entries]
        print(f'Mode: annotated   sequences: {len(work_items)}')
    else:
        work_items = [(seq, None, None) for seq in seqs]
        print(f'Mode: unsupervised   sequences: {len(work_items)}')

    rows_csv = []
    for i, (seq, d_xy, g_xy) in enumerate(work_items):
        img_path  = root / 'JPEGImages' / seq / f'{seq}.png'
        fw_path   = flow_dir / seq / f'{seq}_1.npy'
        bw_path   = bw_dir   / seq / f'{seq}_1.npy'
        if not img_path.exists() or not fw_path.exists():
            continue

        img  = Image.open(img_path).convert('RGB')
        W, H = img.size
        fw = np.load(fw_path).astype(np.float32)
        bw = np.load(bw_path).astype(np.float32) if bw_path.exists() \
             else np.zeros_like(fw)

        mag, grad_mag, div, curl, shear, stretch, strain, \
            lap_div, lap_mag, lap_vec, hess_det = flow_maps(fw)
        global_mag = float(mag.mean())

        # ── bidirectional ─────────────────────────────────────────────────
        consist = np.sqrt((fw[...,0]+bw[...,0])**2 + (fw[...,1]+bw[...,1])**2)
        _, _, div_bw, _, _, _, strain_bw, _, _, _, _ = flow_maps(bw)
        div_diff    = div    - div_bw
        strain_diff = strain - strain_bw
        cand_score  = soft_candidate_score(div_diff, strain)

        # ── CSV row (global stats always; point stats only with annotation) ──
        row = {
            'seq': seq, 'img_W': W, 'img_H': H,
            'global_mag':     f'{global_mag:.3f}',
            'cand_mean':      f'{float(cand_score.mean()):.4f}',
            'cand_p95':       f'{float(np.percentile(cand_score, 95)):.4f}',
            'div_neg_frac':   f'{float((div < 0).mean()):.4f}',
            'div_diff_neg_frac': f'{float((div_diff < 0).mean()):.4f}',
            'strain_mean':    f'{float(strain.mean()):.4f}',
            'consist_mean':   f'{float(consist.mean()):.4f}',
        }

        if args.use_ann and d_xy is not None:
            dx, dy = int(d_xy[0]*W), int(d_xy[1]*H)
            gx, gy = int(g_xy[0]*W), int(g_xy[1]*H)

            def lm(arr, cx, cy): return local_mean(arr, cx, cy, SAMPLE_R)
            d_u, d_v, d_mag = local_flow_stats(fw, dx, dy, SAMPLE_R)
            g_u, g_v, g_mag = local_flow_stats(fw, gx, gy, SAMPLE_R)

            row.update({
                'D_x': f'{d_xy[0]:.4f}', 'D_y': f'{d_xy[1]:.4f}',
                'G_x': f'{g_xy[0]:.4f}', 'G_y': f'{g_xy[1]:.4f}',
                'D_mag': f'{d_mag:.3f}',  'G_mag': f'{g_mag:.3f}',
                'D_rank_pct': f'{float((mag < d_mag).mean())*100:.1f}',
                'G_rank_pct': f'{float((mag < g_mag).mean())*100:.1f}',
                'D_cos': f'{radial_cosine(fw, dx, dy):.4f}',
                'G_cos': f'{radial_cosine(fw, gx, gy):.4f}',
                'D_div':     f'{lm(div,     dx,dy):.4f}', 'G_div':     f'{lm(div,     gx,gy):.4f}',
                'D_strain':  f'{lm(strain,  dx,dy):.4f}', 'G_strain':  f'{lm(strain,  gx,gy):.4f}',
                'D_div_diff':f'{lm(div_diff,dx,dy):.4f}', 'G_div_diff':f'{lm(div_diff,gx,gy):.4f}',
                'D_cand':    f'{lm(cand_score,dx,dy):.4f}','G_cand':   f'{lm(cand_score,gx,gy):.4f}',
            })

        rows_csv.append(row)

        if not args.no_vis:
            make_vis(img, fw, bw, out_dir / f'{seq}.png', d_xy=d_xy, g_xy=g_xy)

        if (i+1) % 50 == 0:
            print(f'  {i+1}/{len(work_items)} done')

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader(); w.writerows(rows_csv)

    # ── TXT ───────────────────────────────────────────────────────────────────
    def a(key):
        vals = []
        for r in rows_csv:
            if key in r:
                vals.append(float(r[key]))
        return np.array(vals) if vals else np.array([float('nan')])

    def gstats(key):
        v = a(key)
        return float(np.nanmean(v)), float(np.nanmedian(v)), float(np.nanstd(v))

    SEP = '=' * 72
    def grow(label, key):
        m, med, s = gstats(key)
        return f'  {label:<26}  mean={m:>8.4f}  median={med:>8.4f}  std={s:>7.4f}'

    mode_str = 'annotated (D+G points)' if args.use_ann else 'unsupervised (all seqs)'
    lines = [
        SEP,
        f'CMC_grasp0 — Flow Field Analysis   mode={mode_str}',
        f'N={len(rows_csv)}   local_r={SAMPLE_R}px',
        SEP,
        '  Global statistics (per-sequence mean, then aggregated across sequences):',
        '  ' + '-' * 68,
        grow('global_mag',        'global_mag'),
        grow('strain_mean',       'strain_mean'),
        grow('consist_mean',      'consist_mean'),
        grow('div_neg_frac',      'div_neg_frac'),
        grow('div_diff_neg_frac', 'div_diff_neg_frac'),
        grow('cand_mean',         'cand_mean'),
        grow('cand_p95',          'cand_p95'),
        '',
    ]

    if args.use_ann:
        def drow(label, kd, kg):
            dm, dmed, ds = gstats(kd)
            gm, gmed, gs = gstats(kg)
            return (f'  {label:<18}  D: {dm:>8.3f}/{dmed:>8.3f}/{ds:>7.3f}'
                    f'    G: {gm:>8.3f}/{gmed:>8.3f}/{gs:>7.3f}')
        lines += [
            '  Point statistics (D=dissect, G=grasp):',
            f'  {"Metric":<18}  {"mean/median/std (D)":^32}    {"mean/median/std (G)":^32}',
            '  ' + '-' * 68,
            drow('mag (local)',  'D_mag',      'G_mag'),
            drow('rank_pct',    'D_rank_pct', 'G_rank_pct'),
            drow('cos (radial)','D_cos',      'G_cos'),
            drow('divergence',  'D_div',      'G_div'),
            drow('strain',      'D_strain',   'G_strain'),
            drow('div_diff',    'D_div_diff', 'G_div_diff'),
            drow('cand_score',  'D_cand',     'G_cand'),
            '',
        ]

    lines += [
        'Sign conventions:',
        '  div      > 0 expand   < 0 shrink/converge',
        '  div_diff = div_fw - div_bw: strongly negative = real convergence sink',
        '  cand     = |div_diff| * strain: high = strong fw/bw asymmetry + deformation = likely soft tissue',
        SEP,
    ]

    txt_path = out_dir / 'flow_stats.txt'
    txt_path.write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines))
    print(f'\nCSV → {csv_path}\nTXT → {txt_path}')


if __name__ == '__main__':
    main()

"""
visualize_cluster_comparison.py

Side-by-side comparison of K-means clustering with different feature weights.

Columns per sample:
  Original | Flow viz | Pure flow | flow+color(1:1) | flow+color(1:2) | flow+color(1:3) | Pure color

Usage:
    python tools/visualize_cluster_comparison.py \
        --n_samples 8 --K 5 --out tools/cluster_comparison.png
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "RAFT" / "core" / "utils"))
from flow_viz import flow_to_image  # noqa: E402


# ── K-means: same logic as training _gpu_kmeans (numpy/CPU version) ──────────

def _train_kmeans(pts: np.ndarray, K: int, n_iter: int = 10) -> np.ndarray:
    """
    Mirrors _gpu_kmeans in models/rcf_soft_tissue_model.py exactly.
    Data-adaptive deterministic init:
      centroid 0   — mean of near-zero pixels (r < 5% max) → background
      centroid 1..K-1 — mean of pixels in each of K-1 equal angular sectors
    Then n_iter=10 EM steps (same as training).
    """
    M, _ = pts.shape
    eps = 1e-6

    r = np.linalg.norm(pts, axis=1)          # [M]
    r_max = r.max() + eps

    # centroid 0: background (near-zero flow)
    bg_mask = r < 0.05 * r_max
    c0 = pts[bg_mask].mean(axis=0) if bg_mask.any() else np.zeros(pts.shape[1])

    # centroids 1..K-1: one per equal angular sector
    n_dir = max(K - 1, 1)
    sector = 2.0 * math.pi / n_dir
    theta = np.arctan2(pts[:, 1], pts[:, 0])   # [-π, π], uses only first 2 dims
    rest_c = []
    for k in range(n_dir):
        angle_k = -math.pi + (k + 0.5) * sector
        diff = theta - angle_k
        diff = diff - (2.0 * math.pi) * np.round(diff / (2.0 * math.pi))
        in_sec = np.abs(diff) < sector * 0.5
        if in_sec.any():
            rest_c.append(pts[in_sec].mean(axis=0))
        else:
            med_r = np.median(r)
            c = np.zeros(pts.shape[1])
            c[0] = med_r * math.cos(angle_k)
            c[1] = med_r * math.sin(angle_k)
            rest_c.append(c)

    centroids = np.stack([c0] + rest_c)        # [K, D]

    # EM: up to n_iter steps, stop early if labels stop changing
    labels = np.zeros(M, dtype=np.int64)
    for _ in range(n_iter):
        diff = pts[:, None, :] - centroids[None, :, :]   # [M, K, D]
        dists = np.linalg.norm(diff, axis=2)              # [M, K]
        new_labels = dists.argmin(axis=1)                 # [M]
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                centroids[k] = pts[mask].mean(axis=0)

    return labels


# ── Features ─────────────────────────────────────────────────────────────────

def flow_features(flow: np.ndarray) -> np.ndarray:
    """(H,W,2) → (N,2) normalised to [-1,1]^2"""
    rad_max = np.linalg.norm(flow, axis=2).max() + 1e-6
    return (flow / rad_max).reshape(-1, 2)


def color_features(img_bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) BGR → (N,3) [cos(2πH), sin(2πH), S]"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    angle = hsv[:, :, 0] / 180.0 * math.pi
    S = hsv[:, :, 1] / 255.0
    return np.stack([np.cos(2 * angle), np.sin(2 * angle), S], axis=-1).reshape(-1, 3)


def cluster_img(labels: np.ndarray, H: int, W: int, K: int) -> np.ndarray:
    cmap = plt.colormaps.get_cmap("tab10").resampled(K)
    return (cmap(labels)[:, :3] * 255).astype(np.uint8).reshape(H, W, 3)


def kmeans(pts: np.ndarray, K: int, n_iter: int = 10) -> np.ndarray:
    return _train_kmeans(pts, K=K, n_iter=n_iter)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sample(data_root: Path, line: str):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    seq_dir = data_root / parts[0]
    img_path = seq_dir / parts[1]
    seq_name = seq_dir.name
    flow_path = data_root / "Flows_NewCT" / seq_name / f"{seq_name}_1.npy"
    if not img_path.exists() or not flow_path.exists():
        imgs = sorted(seq_dir.glob("*.png"))
        flows = sorted((data_root / "Flows_NewCT" / seq_name).glob("*.npy"))
        if not imgs or not flows:
            return None
        img_path, flow_path = imgs[0], flows[0]
    img = cv2.imread(str(img_path))
    flow = np.load(str(flow_path)).astype(np.float32)
    if img is None:
        return None
    if flow.shape[:2] != img.shape[:2]:
        flow = cv2.resize(flow, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img, flow


# ── Main ─────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (title,               fw,  cw,  n_iter)
    ("Pure flow\n10 iter\n(training)",  1.0, 0.0,  10),
    ("Pure flow\nconverged",            1.0, 0.0, 100),
    ("fw=1 cw=2\n10 iter",             1.0, 2.0,  10),
    ("fw=1 cw=2\nconverged",           1.0, 2.0, 100),
    ("Pure color\nconverged",          0.0, 1.0, 100),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=(
        "/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced"))
    ap.add_argument("--split", default="ImageSets/trainval.txt")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out", default="tools/cluster_comparison.png")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    lines = [l for l in (data_root / args.split).read_text().splitlines() if l.strip()]
    step = max(1, len(lines) // args.n_samples)
    selected = lines[::step][: args.n_samples]

    n_extra = 2  # Original + Flow viz
    n_cols = n_extra + len(CONFIGS)
    n_rows = len(selected)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.8))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Original", "Flow\n(RAFT color)"] + [c[0] for c in CONFIGS]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight="bold")

    for row, line in enumerate(selected):
        result = load_sample(data_root, line)
        if result is None:
            continue
        img, flow = result
        H, W = img.shape[:2]

        f_pts = flow_features(flow)   # (N, 2)
        c_pts = color_features(img)   # (N, 3)

        # Col 0: original
        axes[row][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Col 1: flow viz
        axes[row][1].imshow(flow_to_image(flow))

        # Remaining cols: clustering configs
        for ci, (_, fw, cw, n_iter) in enumerate(CONFIGS):
            parts_list = []
            if fw > 0:
                parts_list.append(f_pts * fw)
            if cw > 0:
                parts_list.append(c_pts * cw)
            pts = np.concatenate(parts_list, axis=1)
            labels = kmeans(pts, K=args.K, n_iter=n_iter)
            axes[row][n_extra + ci].imshow(cluster_img(labels, H, W, args.K))

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"K-means cluster comparison  |  K={args.K}", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

"""
visualize_joint_clusters.py

Compare three K-means clustering strategies on grasp0 samples:
  Col 1 — Original RGB frame
  Col 2 — RAFT flow_to_color visualisation
  Col 3 — Flow-only K-means clusters   (existing approach)
  Col 4 — Joint (flow + HSV) K-means clusters  (proposed)

Usage:
    python tools/visualize_joint_clusters.py \
        --data_root /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced \
        --n_samples 12 \
        --K 5 \
        --flow_weight 1.0 \
        --color_weight 1.0 \
        --out tools/joint_cluster_vis.png
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent / "RAFT" / "core" / "utils"))
from flow_viz import flow_to_image  # noqa: E402


# ── K-means helpers ──────────────────────────────────────────────────────────

def kmeans_labels(pts: np.ndarray, K: int, n_init: int = 3) -> np.ndarray:
    """Run K-means on (N, D) feature array, return (N,) int labels."""
    km = KMeans(n_clusters=K, n_init=n_init, random_state=0)
    return km.fit_predict(pts)


def flow_features(flow: np.ndarray) -> np.ndarray:
    """
    flow: (H, W, 2) raw RAFT flow
    Returns (H*W, 2) normalised to [-1, 1]^2, same space as flow_to_color.
    Centroid 0 seeded at origin (background), rest at equal angular sectors.
    """
    H, W, _ = flow.shape
    rad = np.linalg.norm(flow, axis=2)          # (H, W)
    rad_max = rad.max() + 1e-6
    pts = (flow / rad_max).reshape(-1, 2)        # (N, 2)
    return pts


def hsv_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    img_bgr: (H, W, 3) uint8 BGR image
    Returns (H*W, 3): [cos(2π*H), sin(2π*H), S]
    Hue encoded as circular unit vector to avoid 0°/360° discontinuity.
    V dropped — brightness irrelevant for colour-block identity.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_norm = img_hsv[:, :, 0] / 180.0 * math.pi   # [0, π] → angle in [0, 2π]
    S_norm = img_hsv[:, :, 1] / 255.0              # [0, 1]
    cos_h = np.cos(2 * H_norm)
    sin_h = np.sin(2 * H_norm)
    pts = np.stack([cos_h, sin_h, S_norm], axis=-1).reshape(-1, 3)
    return pts


def cluster_colormap(labels: np.ndarray, H: int, W: int, K: int) -> np.ndarray:
    """Map integer cluster labels to an RGB image."""
    cmap = plt.colormaps.get_cmap("tab10").resampled(K)
    rgb = (cmap(labels)[:, :3] * 255).astype(np.uint8)
    return rgb.reshape(H, W, 3)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sample(data_root: Path, line: str):
    """
    trainval.txt line format:
      JPEGImages/{seq}/ {seq}.png {seq}_1.png
    Returns (img_bgr, flow) or None on error.
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    seq_dir = data_root / parts[0]
    img_path = seq_dir / parts[1]
    # flow: {seq}_1.npy (forward flow from frame0 to frame1)
    seq_name = seq_dir.name
    flow_path = data_root / "Flows_NewCT" / seq_name / f"{seq_name}_1.npy"
    if not img_path.exists() or not flow_path.exists():
        imgs = sorted(seq_dir.glob("*.png"))
        flows = sorted((data_root / "Flows_NewCT" / seq_name).glob("*.npy"))
        if not imgs or not flows:
            return None
        img_path = imgs[0]
        flow_path = flows[0]

    img = cv2.imread(str(img_path))
    flow = np.load(str(flow_path)).astype(np.float32)   # (H, W, 2)
    if img is None:
        return None
    # resize flow to image size if needed
    if flow.shape[:2] != img.shape[:2]:
        flow = cv2.resize(flow, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return img, flow


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=(
        "/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced"))
    ap.add_argument("--split", default="ImageSets/trainval.txt")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--flow_weight", type=float, default=1.0,
                    help="Weight on normalised flow features in joint clustering")
    ap.add_argument("--color_weight", type=float, default=1.0,
                    help="Weight on HSV features in joint clustering")
    ap.add_argument("--out", default="tools/joint_cluster_vis.png")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    split_path = data_root / args.split
    lines = [l for l in split_path.read_text().splitlines() if l.strip()]

    # pick evenly spaced samples
    step = max(1, len(lines) // args.n_samples)
    selected = lines[::step][: args.n_samples]

    n_cols = 4
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Original", "Flow (RAFT color)", "Flow K-means", "Joint K-means\n(flow + HSV)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row, line in enumerate(selected):
        result = load_sample(data_root, line)
        if result is None:
            print(f"[WARN] Could not load sample: {line}")
            continue
        img_bgr, flow = result
        H, W = img_bgr.shape[:2]

        # ── Col 0: original RGB ───────────────────────────────────────────
        axes[row][0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ── Col 1: flow_to_color ──────────────────────────────────────────
        flow_vis = flow_to_image(flow)                      # (H, W, 3) RGB
        axes[row][1].imshow(flow_vis)

        # ── Col 2: flow-only K-means ──────────────────────────────────────
        f_pts = flow_features(flow)                         # (N, 2)
        f_labels = kmeans_labels(f_pts, K=args.K)
        axes[row][2].imshow(cluster_colormap(f_labels, H, W, args.K))

        # ── Col 3: joint K-means (flow + HSV) ────────────────────────────
        c_pts = hsv_features(img_bgr)                       # (N, 3)
        joint_pts = np.concatenate([
            f_pts * args.flow_weight,
            c_pts * args.color_weight,
        ], axis=1)                                          # (N, 5)
        j_labels = kmeans_labels(joint_pts, K=args.K)
        axes[row][3].imshow(cluster_colormap(j_labels, H, W, args.K))

        for ax in axes[row]:
            ax.axis("off")

        seq_name = line.split("/")[1] if "/" in line else line
        axes[row][0].set_ylabel(seq_name[:20], fontsize=7, rotation=0,
                                labelpad=60, va="center")

    plt.suptitle(
        f"Joint clustering  |  K={args.K}  "
        f"flow_weight={args.flow_weight}  color_weight={args.color_weight}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

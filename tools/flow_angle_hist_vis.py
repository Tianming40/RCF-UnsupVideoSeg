"""
Compare 1D angle k-means: pixel-count weight vs magnitude weight.

Layout (2 rows × 3 cols):
  Row 0: original | flow_vis | angle colormap
  Row 1: pixel-count hist + overlay | magnitude hist + overlay | side-by-side comparison

Usage:
    python tools/flow_angle_hist_vis.py --n 8 --out tools/flow_angle_vis_cmp
"""

import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
import flow_vis


K = 5
COLORS = np.array([
    [220,  50,  50],
    [ 50, 180,  50],
    [ 50, 100, 220],
    [220, 180,  50],
    [180,  50, 200],
], dtype=np.float32) / 255.0


def load_sample(seq_dir, jpeg_dir):
    flows = sorted(Path(seq_dir).glob("*_1.npy"))
    if not flows:
        return None
    flow_path = flows[0]
    stem = flow_path.stem.replace("_1", "")
    img_path = Path(jpeg_dir) / stem / f"{stem}.png"
    if not img_path.exists():
        return None
    flow = np.load(flow_path).astype(np.float32)
    img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    return img, flow, str(flow_path)


def flow_angle_raft(flow):
    return np.arctan2(-flow[..., 1], -flow[..., 0])


def flow_magnitude(flow):
    return np.linalg.norm(flow, axis=-1)


def angle_to_rgb(angle):
    hue = (angle + np.pi) / (2 * np.pi)
    H, W = hue.shape
    hsv = np.stack([hue, np.ones((H, W)), np.ones((H, W))], axis=-1).astype(np.float32)
    return matplotlib.colors.hsv_to_rgb(hsv)


def make_overlay(img, cluster_map):
    base = img.astype(np.float32) / 255.0
    out  = base.copy()
    for c in range(K):
        m = cluster_map == c
        out[m] = 0.5 * base[m] + 0.5 * COLORS[c]
    return out


def circular_kmeans(angles, weights, k=K, n_init=5, max_iter=50):
    """Weighted circular k-means. Pass weights=ones for pixel-count mode."""
    best_loss, best_labels, best_centers = np.inf, None, None
    for trial in range(n_init):
        centers = (np.linspace(-np.pi, np.pi, k, endpoint=False) if trial == 0
                   else np.sort(np.random.choice(angles, k, replace=False,
                                                  p=weights / weights.sum())))
        for _ in range(max_iter):
            d = np.abs(angles[:, None] - centers[None, :])
            d = np.minimum(d, 2 * np.pi - d)
            labels = d.argmin(axis=1)
            new_c = np.zeros(k)
            for c in range(k):
                m = labels == c
                if not m.any():
                    new_c[c] = centers[c]; continue
                w, a = weights[m], angles[m]
                new_c[c] = np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())
            if np.allclose(new_c, centers, atol=1e-4):
                break
            centers = np.sort(new_c)
        d = np.abs(angles[:, None] - centers[None, :])
        d = np.minimum(d, 2 * np.pi - d)
        labels = d.argmin(axis=1)
        loss = (weights * d[np.arange(len(angles)), labels] ** 2).sum()
        if loss < best_loss:
            best_loss, best_labels, best_centers = loss, labels.copy(), centers.copy()
    return best_labels, best_centers


def visualize_sample(img, flow, title, out_path):
    H, W = flow.shape[:2]
    angle    = flow_angle_raft(flow)
    mag      = flow_magnitude(flow)
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    angle_rgb = angle_to_rgb(angle)

    n_bins    = 180
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_ctr   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    thresh   = mag > mag.max() * 0.05
    flat_idx = np.where(thresh.ravel())[0]
    ang_flat = angle[thresh].ravel()
    mag_flat = mag[thresh].ravel()
    one_flat = np.ones_like(mag_flat)   # unit weight = pixel count

    hist_count, _ = np.histogram(ang_flat, bins=bin_edges)
    hist_mag,   _ = np.histogram(ang_flat, bins=bin_edges, weights=mag_flat)

    labels_cnt, centers_cnt = circular_kmeans(ang_flat, one_flat)
    labels_mag, centers_mag = circular_kmeans(ang_flat, mag_flat)

    def to_map(labels):
        m = np.full(H * W, -1, dtype=np.int32)
        m[flat_idx] = labels
        return m.reshape(H, W)

    cmap_cnt = to_map(labels_cnt)
    cmap_mag = to_map(labels_mag)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=10)

    # Row 0
    axes[0, 0].imshow(img);       axes[0, 0].set_title("Original");      axes[0, 0].axis('off')
    axes[0, 1].imshow(flow_rgb);  axes[0, 1].set_title("flow_vis");       axes[0, 1].axis('off')
    axes[0, 2].imshow(angle_rgb); axes[0, 2].set_title("Angle colormap"); axes[0, 2].axis('off')

    # Row 1 col 0 — pixel count histogram + overlay
    ax = axes[1, 0]
    ax.bar(bin_ctr, hist_count, width=2*np.pi/n_bins, color='orange', alpha=0.7, label='pixel count')
    sort_cnt = np.argsort(centers_cnt)
    for i, c in enumerate(centers_cnt[sort_cnt]):
        ax.axvline(c, color=COLORS[sort_cnt[i]], lw=2, label=f"{np.degrees(c):.0f}°")
    ax.set_title("Pixel-count histogram + k-means centers")
    ax.legend(fontsize=6); ax.set_xlim(-np.pi, np.pi)

    # Row 1 col 1 — magnitude histogram + overlay
    ax = axes[1, 1]
    ax.bar(bin_ctr, hist_mag, width=2*np.pi/n_bins, color='steelblue', alpha=0.7, label='magnitude sum')
    sort_mag = np.argsort(centers_mag)
    for i, c in enumerate(centers_mag[sort_mag]):
        ax.axvline(c, color=COLORS[sort_mag[i]], lw=2, label=f"{np.degrees(c):.0f}°")
    ax.set_title("Magnitude histogram + k-means centers")
    ax.legend(fontsize=6); ax.set_xlim(-np.pi, np.pi)

    # Row 1 col 2 — side-by-side overlay
    combined = np.concatenate([make_overlay(img, cmap_cnt), make_overlay(img, cmap_mag)], axis=1)
    axes[1, 2].imshow(combined)
    axes[1, 2].set_title("Left: pixel-count k-means    Right: magnitude k-means")
    axes[1, 2].axis('off')
    # dividing line
    axes[1, 2].axvline(W, color='white', lw=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
    parser.add_argument('--n',    type=int, default=8)
    parser.add_argument('--out',  default='tools/flow_angle_vis_cmp')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    flow_root = Path(args.data) / 'Flows_NewCT'
    jpeg_root = Path(args.data) / 'JPEGImages'
    seqs = sorted(flow_root.iterdir())
    random.shuffle(seqs)
    seqs = seqs[:args.n * 3]

    count = 0
    for seq in seqs:
        if count >= args.n: break
        result = load_sample(seq, jpeg_root)
        if result is None: continue
        img, flow, path = result
        title = Path(path).parent.name
        visualize_sample(img, flow, title, os.path.join(args.out, f"{title}.png"))
        count += 1

    print(f"\nDone. {count} images → {args.out}")


if __name__ == '__main__':
    main()

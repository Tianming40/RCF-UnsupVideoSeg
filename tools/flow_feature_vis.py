"""
Compare K=4 clustering with different flow feature combinations.

Layout (2 rows x 4 cols):
  Row 0: original | flow_vis | angle colormap | magnitude
  Row 1: kmeans(angle) | kmeans(angle+xy) | kmeans(angle+mag+xy) | divergence+curl

Usage:
    python tools/flow_feature_vis.py --n 8 --out tools/flow_feature_vis_out
"""

import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import flow_vis

K = 4
COLORS = np.array([
    [220,  50,  50],
    [ 50, 180,  50],
    [ 50, 100, 220],
    [220, 180,  50],
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


def flow_features(flow):
    H, W = flow.shape[:2]
    vx, vy = flow[..., 0], flow[..., 1]

    angle = np.arctan2(-vy, -vx)           # RAFT convention
    mag   = np.sqrt(vx**2 + vy**2)

    # divergence and curl via finite differences
    dvx_dx = np.gradient(vx, axis=1)
    dvy_dy = np.gradient(vy, axis=0)
    dvy_dx = np.gradient(vy, axis=1)
    dvx_dy = np.gradient(vx, axis=0)
    div  = dvx_dx + dvy_dy
    curl = dvy_dx - dvx_dy

    # pixel coordinates normalized to [0,1]
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    xmap, ymap = np.meshgrid(xs, ys)

    return angle, mag, div, curl, xmap, ymap


def kmeans_fit(feat, k=K, n_init=5, max_iter=50):
    """feat: [N, D] float32. Returns labels [N]."""
    N, D = feat.shape
    best_loss, best_labels = np.inf, None
    for _ in range(n_init):
        idx = np.random.choice(N, k, replace=False)
        centers = feat[idx].copy()
        for _ in range(max_iter):
            dists  = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [N,K]
            labels = dists.argmin(axis=1)
            new_c  = np.array([feat[labels == c].mean(0) if (labels == c).any()
                                else centers[c] for c in range(k)])
            if np.allclose(new_c, centers, atol=1e-5):
                break
            centers = new_c
        dists  = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(axis=1)
        loss   = dists[np.arange(N), labels].sum()
        if loss < best_loss:
            best_loss, best_labels = loss, labels.copy()
    return best_labels


def make_overlay(img, label_map, mask=None):
    base = img.astype(np.float32) / 255.0
    out  = base.copy()
    for c in range(K):
        m = label_map == c
        if mask is not None:
            m = m & mask
        out[m] = 0.5 * base[m] + 0.5 * COLORS[c]
    return out


def angle_to_rgb(angle):
    hue = (angle + np.pi) / (2 * np.pi)
    H, W = hue.shape
    hsv = np.stack([hue, np.ones((H, W)), np.ones((H, W))], axis=-1).astype(np.float32)
    return matplotlib.colors.hsv_to_rgb(hsv)


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def visualize_sample(img, flow, title, out_path):
    H, W = flow.shape[:2]
    angle, mag, div, curl, xmap, ymap = flow_features(flow)
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)

    # magnitude threshold mask
    thresh = mag > mag.max() * 0.05
    flat   = thresh.ravel()
    idx    = np.where(flat)[0]

    def flat_feat(*arrs):
        return np.stack([a.ravel()[idx] for a in arrs], axis=1).astype(np.float32)

    def to_map(labels):
        m = np.full(H * W, -1, dtype=np.int32)
        m[idx] = labels
        return m.reshape(H, W)

    # Feature sets
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    mag_n = norm01(mag)

    feat_mag       = flat_feat(mag_n)
    feat_mag_xy    = flat_feat(mag_n, xmap * 0.5, ymap * 0.5)
    feat_angle_xy  = flat_feat(cos_a, sin_a, xmap * 0.5, ymap * 0.5)
    feat_all       = flat_feat(cos_a, sin_a, mag_n * 0.3, xmap * 0.5, ymap * 0.5)

    lbl_mag       = to_map(kmeans_fit(feat_mag))
    lbl_mag_xy    = to_map(kmeans_fit(feat_mag_xy))
    lbl_angle_xy  = to_map(kmeans_fit(feat_angle_xy))
    lbl_all       = to_map(kmeans_fit(feat_all))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(title, fontsize=9)

    # Row 0
    axes[0, 0].imshow(img);                     axes[0, 0].set_title("Original")
    axes[0, 1].imshow(flow_rgb);                axes[0, 1].set_title("flow_vis (RAFT)")
    axes[0, 2].imshow(angle_to_rgb(angle));     axes[0, 2].set_title("Angle")
    axes[0, 3].imshow(norm01(mag), cmap='hot'); axes[0, 3].set_title("Magnitude")

    # Row 1
    axes[1, 0].imshow(make_overlay(img, lbl_mag,      thresh)); axes[1, 0].set_title("K=4: mag only")
    axes[1, 1].imshow(make_overlay(img, lbl_mag_xy,   thresh)); axes[1, 1].set_title("K=4: mag + xy")
    axes[1, 2].imshow(make_overlay(img, lbl_angle_xy, thresh)); axes[1, 2].set_title("K=4: angle + xy")
    axes[1, 3].imshow(make_overlay(img, lbl_all,      thresh)); axes[1, 3].set_title("K=4: angle + mag + xy")

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
    parser.add_argument('--n',    type=int, default=8)
    parser.add_argument('--out',  default='tools/flow_feature_vis_out')
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

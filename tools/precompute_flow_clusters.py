"""
Precompute K-means flow cluster labels for all flow files in a dataset.

Saves cluster labels (uint8) alongside flow files:
  Flows_NewCT/{seq}/{name}.npy         → FlowClusters_NewCT/{seq}/{name}.npy
  BackwardFlows_NewCT/{seq}/{name}.npy → BackwardFlowClusters_NewCT/{seq}/{name}.npy

Usage:
  python tools/precompute_flow_clusters.py \
      --dataset /path/to/dataset \
      [--dataset /path/to/dataset2 ...] \
      --K 5 --n_iter 50 --gpu 0

The output cluster label files are uint8 arrays of shape (H, W).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tqdm import tqdm

# K distinct colors for cluster visualization (up to 10 clusters)
_CLUSTER_COLORS = np.array([
    [180, 180, 180],   # 0 — gray        (stationary / background)
    [ 76, 153, 255],   # 1 — blue
    [ 76, 204,  76],   # 2 — green
    [255, 102,  76],   # 3 — orange
    [204,  76, 204],   # 4 — purple
    [255, 220,  50],   # 5 — yellow
    [ 50, 210, 210],   # 6 — cyan
    [255, 100, 180],   # 7 — pink
    [130,  80,  40],   # 8 — brown
    [ 80, 130,  50],   # 9 — dark green
], dtype=np.uint8)


# ── GPU K-means — exact copy of _gpu_kmeans in rcf_soft_tissue_model.py ────
# Init: centroid 0 = origin (stationary bg); others = evenly on ring r=0.5.
# This is deterministic, so precomputed labels match training exactly.

def gpu_kmeans(pts: torch.Tensor, K: int, n_iter: int) -> torch.Tensor:
    """K-means on GPU. pts: (M, 2). Returns labels (M,) int64."""
    M, _ = pts.shape
    device = pts.device

    angles = torch.linspace(
        0, 2 * 3.14159265 * (1 - 1 / max(K - 1, 1)), max(K - 1, 1), device=device
    )
    ring = torch.stack([0.5 * torch.cos(angles), 0.5 * torch.sin(angles)], dim=1)
    origin = torch.zeros(1, 2, device=device)
    centroids = torch.cat([origin, ring[:K - 1]], dim=0)  # (K, 2)

    labels = torch.zeros(M, dtype=torch.long, device=device)
    for _ in range(n_iter):
        dists = (pts.unsqueeze(1) - centroids.unsqueeze(0)).norm(dim=2)  # (M, K)
        labels = dists.argmin(dim=1)                                       # (M,)
        one_hot = torch.zeros(M, K, device=device).scatter_(1, labels.unsqueeze(1), 1)
        counts = one_hot.sum(0).clamp(min=1)
        centroids = (one_hot.T @ pts) / counts[:, None]

    return labels


def cluster_flow_file(
    flow_path: Path,
    out_path: Path,
    K: int,
    n_iter: int,
    device: torch.device,
    eps: float = 1e-6,
):
    """Load one flow .npy, run K-means, save cluster labels + color visualization."""
    flow = np.load(flow_path).astype(np.float32)   # (H, W, 2) HWC

    flow_t = torch.from_numpy(flow).to(device)     # (H, W, 2)
    H, W, _ = flow_t.shape

    # normalize by global max radius (mirrors flow_to_color convention)
    rad_max = flow_t.norm(dim=-1).max().clamp(min=eps)
    fn = flow_t / rad_max                          # (H, W, 2)
    pts = fn.reshape(-1, 2)                        # (H*W, 2)

    with torch.no_grad():
        labels = gpu_kmeans(pts, K=K, n_iter=n_iter)   # (H*W,)

    labels_np = labels.cpu().numpy().reshape(H, W).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, labels_np)

    # color visualization — same resolution as original flow (H, W)
    colors = _CLUSTER_COLORS[:K]
    vis = colors[labels_np]                        # (H, W, 3) uint8
    vis_path = out_path.with_suffix('.png')
    Image.fromarray(vis).save(vis_path)


def process_flow_dir(
    flow_dir: Path,
    out_dir: Path,
    K: int,
    n_iter: int,
    device: torch.device,
    skip_existing: bool = True,
):
    files = sorted(flow_dir.rglob("*.npy"))
    if not files:
        print(f"  [skip] no .npy files in {flow_dir}")
        return

    skipped = 0
    for fpath in tqdm(files, desc=str(flow_dir.name), unit="file"):
        rel = fpath.relative_to(flow_dir)
        out_path = out_dir / rel
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        cluster_flow_file(fpath, out_path, K=K, n_iter=n_iter, device=device)

    if skipped:
        print(f"  skipped {skipped} already-computed files")


def main():
    parser = argparse.ArgumentParser(description="Precompute flow cluster labels")
    parser.add_argument("--dataset", action="append", required=True,
                        metavar="PATH", dest="datasets",
                        help="Dataset root (can be repeated for multiple datasets)")
    parser.add_argument("--flow_suffixes", nargs="+",
                        default=["_NewCT"],
                        help="Flow directory suffixes to process (default: _NewCT)")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of K-means clusters (default: 5)")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="K-means iterations (default: 50)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id (default: 0)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Recompute even if output file exists")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  K={args.K}  n_iter={args.n_iter}")

    for dataset_root in args.datasets:
        root = Path(dataset_root)
        print(f"\nDataset: {root}")

        for suffix in args.flow_suffixes:
            for prefix in ("Flows", "BackwardFlows"):
                flow_dir = root / f"{prefix}{suffix}"
                out_dir  = root / f"{prefix}Clusters{suffix}"
                if not flow_dir.exists():
                    print(f"  [skip] {flow_dir} not found")
                    continue
                print(f"  {flow_dir.name}  →  {out_dir.name}")
                process_flow_dir(
                    flow_dir, out_dir,
                    K=args.K, n_iter=args.n_iter, device=device,
                    skip_existing=not args.no_skip,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()

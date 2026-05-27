#!/usr/bin/env python3
"""
Post-process optical flow .npy files by subtracting the per-channel spatial mean
(camera motion compensation / global translation removal).

  flow_nm[c] = flow[c] - mean(flow[c])   for c in {u, v}

This removes the global translation component (camera shake) so that only
object-relative motion remains. Same approach as FlowDINO (IROS 2023).

Input:
  CMC_grasp10_deinterlaced/Flows_NewCT/<STEM>/<STEM>_1.npy
  CMC_grasp10_deinterlaced/BackwardFlows_NewCT/<STEM>/<STEM>_1.npy

Output:
  CMC_grasp10_deinterlaced/Flows_NewCT_nm/<STEM>/<STEM>_1.npy
  CMC_grasp10_deinterlaced/BackwardFlows_NewCT_nm/<STEM>/<STEM>_1.npy

Config change needed after running:
  flow_suffix: "_NewCT_nm"

Usage:
  python tools/normalize_flows.py
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, 'RAFT/core')
from utils import flow_viz
HAS_VIZ = True

DATA_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced")

PAIRS = [
    ("Flows_NewCT",         "Flows_NewCT_nm"),
    ("BackwardFlows_NewCT", "BackwardFlows_NewCT_nm"),
]


def subtract_mean(flow: np.ndarray) -> np.ndarray:
    """
    flow: (H, W, 2) float32
    Returns flow with global mean subtracted per channel.
    Same as FlowDINO: flow - flow.reshape(2,-1).mean(axis=1)[:,None,None]
    """
    # Transpose to (2, H, W) for easier mean computation
    f = flow.transpose(2, 0, 1)          # (2, H, W)
    mean_u = f[0].mean()
    mean_v = f[1].mean()
    f[0] -= mean_u
    f[1] -= mean_v
    return f.transpose(1, 2, 0)          # back to (H, W, 2)


def main():
    for src_name, dst_name in PAIRS:
        src_root = DATA_ROOT / src_name
        dst_root = DATA_ROOT / dst_name
        dst_root.mkdir(exist_ok=True)

        all_npy = sorted(src_root.glob("*/*.npy"))
        print(f"\n{src_name} → {dst_name}  ({len(all_npy)} files)")

        skipped = 0
        stats_before = []
        stats_after  = []

        for src_path in tqdm(all_npy, desc=dst_name):
            rel      = src_path.relative_to(src_root)   # <STEM>/<STEM>_1.npy
            dst_path = dst_root / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            png_path = dst_path.with_suffix('.png')
            npy_exists = dst_path.exists()
            png_exists = png_path.exists()

            if npy_exists and png_exists:
                skipped += 1
                continue

            if npy_exists:
                # npy already done, only regenerate PNG
                flow_nm = np.load(dst_path).astype(np.float32)
                mag_before = mag_after = 0.
            else:
                flow = np.load(src_path).astype(np.float32)   # (H, W, 2)
                mag_before = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                flow_nm = subtract_mean(flow)
                mag_after = np.sqrt(flow_nm[..., 0]**2 + flow_nm[..., 1]**2).mean()
                np.save(str(dst_path), flow_nm.astype(np.float16))

            # Save visualisation PNG
            if HAS_VIZ and not png_exists:
                cv2.imwrite(str(png_path), flow_viz.flow_to_image(flow_nm))

            stats_before.append(mag_before)
            stats_after.append(mag_after)

        n = len(stats_before)
        if n > 0:
            print(f"  Processed: {n}   Skipped (already exist): {skipped}")
            print(f"  Mean flow magnitude  before: {np.mean(stats_before):.2f} px")
            print(f"  Mean flow magnitude  after:  {np.mean(stats_after):.2f} px")
        else:
            print(f"  All {skipped} files already existed, nothing to do.")

    print("\nDone.")
    print("Update your training config:")
    print("  flow_suffix:  \"_NewCT_nm\"")


if __name__ == "__main__":
    main()

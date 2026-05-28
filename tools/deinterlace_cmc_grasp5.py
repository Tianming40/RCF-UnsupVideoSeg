#!/usr/bin/env python3
"""
Bob deinterlacing for CMC grasp-5 images (same field order as grasp-10).

Even rows (0,2,4...) = OLD field (T-0.5) → copy from odd row below.
Odd rows  (1,3,5...) = NEW field (T)     → keep.

Input:  CMC_grasp5_from_raw/JPEGImages/<STEM>/
Output: CMC_grasp5_deinterlaced/JPEGImages/<STEM>/

Usage:
  python tools/deinterlace_cmc_grasp5.py
"""

import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp5_from_raw")
DST_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp5_deinterlaced")


def bob_deinterlace(img_array: np.ndarray) -> np.ndarray:
    """Keep odd rows (new field), copy each to the even row above it."""
    out = img_array.copy()
    out[0::2] = out[1::2]
    return out


def main():
    src_jpeg = SRC_ROOT / "JPEGImages"
    all_pngs = sorted(src_jpeg.glob("*/*.png"))
    print(f"Found {len(all_pngs)} images under {src_jpeg}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "Flows_NewCT").mkdir(exist_ok=True)
    (DST_ROOT / "BackwardFlows_NewCT").mkdir(exist_ok=True)

    skipped = 0
    for src_path in tqdm(all_pngs, desc="Deinterlacing"):
        rel = src_path.relative_to(src_jpeg)
        dst_path = DST_ROOT / "JPEGImages" / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            skipped += 1
            continue

        img = np.array(Image.open(src_path).convert("RGB"))
        img_out = bob_deinterlace(img)
        Image.fromarray(img_out).save(dst_path)

    print(f"Done. Skipped {skipped} already-existing files.")

    src_imagesets = SRC_ROOT / "ImageSets"
    dst_imagesets = DST_ROOT / "ImageSets"
    dst_imagesets.mkdir(exist_ok=True)
    for txt in src_imagesets.glob("*.txt"):
        dst_txt = dst_imagesets / txt.name
        if not dst_txt.exists():
            shutil.copy2(txt, dst_txt)
            print(f"  Copied ImageSets/{txt.name}")
        else:
            print(f"  Already exists: ImageSets/{txt.name}")

    print(f"\nOutput: {DST_ROOT}")
    print("Next step: bash run_flows_grasp5.sh")


if __name__ == "__main__":
    main()

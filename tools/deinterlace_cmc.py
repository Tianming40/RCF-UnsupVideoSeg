#!/usr/bin/env python3
"""
Bob deinterlacing for CMC grasp-10 images (corrected field order).

Analysis of CMC/grasp-10 via check_interlace.py showed:
  - Even rows (0,2,4...) ≈ PRE frame  →  even rows are the OLD field (T-0.5)
  - Odd rows  (1,3,5...) are the NEW field (current time T)

Correct Bob method: KEEP ODD rows, copy each odd row to the even row above it.

  row 0 (even) → copy from row 1
  row 1 (odd)  → keep
  row 2 (even) → copy from row 3
  row 3 (odd)  → keep
  ...

Input:  CMC_grasp10_from_raw/JPEGImages/<STEM>/  (one subdir per pair)
Output: CMC_grasp10_deinterlaced/JPEGImages/<STEM>/  (same structure, actual files)

ImageSets/ txt files and empty Flows_NewCT/ dirs are also created so the
output dataset is immediately usable by the training pipeline.

Usage:
  python tools/deinterlace_cmc.py
"""

import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_from_raw")
DST_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced")


def bob_deinterlace(img_array: np.ndarray) -> np.ndarray:
    """Keep odd rows (new field), copy each to the even row above it."""
    out = img_array.copy()
    # odd rows:  1, 3, 5, ...  → keep  (current time T)
    # even rows: 0, 2, 4, ...  → copy from the odd row below
    out[0::2] = out[1::2]
    return out


def main():
    # ── Collect all images to process ─────────────────────────────────
    src_jpeg = SRC_ROOT / "JPEGImages"
    all_pngs = sorted(src_jpeg.glob("*/*.png"))
    print(f"Found {len(all_pngs)} images under {src_jpeg}")

    # ── Create output directory skeleton ──────────────────────────────
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "Flows_NewCT").mkdir(exist_ok=True)
    (DST_ROOT / "BackwardFlows_NewCT").mkdir(exist_ok=True)

    # ── Deinterlace each image ─────────────────────────────────────────
    skipped = 0
    for src_path in tqdm(all_pngs, desc="Deinterlacing"):
        # Preserve subdir: JPEGImages/<STEM>/<file>.png
        rel = src_path.relative_to(src_jpeg)          # e.g. 9639.../9639....png
        dst_path = DST_ROOT / "JPEGImages" / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            skipped += 1
            continue

        img = np.array(Image.open(src_path).convert("RGB"))
        img_out = bob_deinterlace(img)
        Image.fromarray(img_out).save(dst_path)

    print(f"Done. Skipped {skipped} already-existing files.")

    # ── Copy ImageSets txt files (paths are relative, no change needed) ─
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
    print("Next step: python RAFT/generate_flows_cmc.py  (update SRC/DST paths to CMC_grasp10_deinterlaced)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bob deinterlacing for CMC_grasp10_medical_format images.

Interlaced frames contain odd rows from frame N and even rows from frame N-1.
Bob method: keep even rows, copy each even row to the odd row below it.

  row 0 (even) → keep
  row 1 (odd)  → copy from row 0
  row 2 (even) → keep
  row 3 (odd)  → copy from row 2
  ...

Output is saved to CMC_grasp10_medical_format_deinterlaced/ (original unchanged).

Usage:
  python tools/deinterlace_cmc.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SRC = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_medical_format/JPEGImages/cmc_sequence")
DST = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_medical_format_deinterlaced/JPEGImages/cmc_sequence")


def bob_deinterlace(img_array: np.ndarray) -> np.ndarray:
    """Keep even rows, copy each to the odd row below."""
    out = img_array.copy()
    # even rows: 0, 2, 4, ...  → keep
    # odd rows:  1, 3, 5, ...  → copy from the even row above
    out[1::2] = out[0::2]
    return out


def main():
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("*.png"))
    print(f"Found {len(files)} images in {SRC}")

    skipped = 0
    for src_path in tqdm(files, desc="Deinterlacing"):
        dst_path = DST / src_path.name
        if dst_path.exists():
            skipped += 1
            continue

        img = np.array(Image.open(src_path).convert("RGB"))
        img_out = bob_deinterlace(img)
        Image.fromarray(img_out).save(dst_path)

    print(f"Done. Skipped {skipped} existing files.")
    print(f"Output: {DST}")
    print("Next step: re-run tools/reorganize_cmc_for_finetune.py pointing SRC_JPEG to the new deinterlaced path.")


if __name__ == "__main__":
    main()

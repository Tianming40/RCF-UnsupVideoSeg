#!/usr/bin/env python3
"""
gen_seg_val.py

Generate a val.txt (test list) under each split annotation folder, single-frame
2-column format (no flow second frame; evaluation is only on the annotated
frame-0):

    JPEGImages/<id>/ <id>.png

Each class list contains only frames whose binary mask is non-empty:
    instrument_seg/val.txt  -> frames that actually contain Instrument
    tissue_seg/val.txt      -> frames that actually contain Soft Tissue

Frames are cross-checked against the main JPEGImages (must have <id>.png); any
missing ones are reported and skipped.
"""

import numpy as np
from PIL import Image
from pathlib import Path

DP = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced")
TM = DP / "tissue mask"
JPEG = DP / "JPEGImages"

CLASSES = [("instrument", "instrument_seg"), ("tissue", "tissue_seg")]

def main():
    for cls, seg in CLASSES:
        seg_dir = TM / seg
        lines, missing, empty = [], [], 0
        for f in sorted(seg_dir.rglob("*.png")):
            if f.name == "val.txt":
                continue
            arr = np.array(Image.open(f).convert("L"))
            if not arr.any():
                empty += 1
                continue
            i = f.stem
            f0 = JPEG / i / f"{i}.png"
            if f0.exists():
                lines.append(f"JPEGImages/{i}/ {i}.png")
            else:
                missing.append(i)
        out = seg_dir / "val.txt"
        out.write_text("\n".join(lines) + "\n")
        print(f"[{cls:10s}] non-empty={len(lines)+len(missing)}  "
              f"written={len(lines)}  empty-skipped={empty}  missing-in-JPEG={len(missing)}")
        if missing:
            print("   missing:", missing)
        print(f"   -> {out}")

if __name__ == "__main__":
    main()

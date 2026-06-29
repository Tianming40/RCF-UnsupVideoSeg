#!/usr/bin/env python3
"""
build_eval_dirs.py

Turn the split per-class annotations into two self-contained eval roots that the
model's val pipeline can consume directly (it locates GT via
  path.replace("JPEGImages", "Annotations").replace(".png", ".jpg")
i.e. <root>/Annotations/<id>/<id>.jpg ).

Produces, under the grasp0 dataset dir:

    eval_instrument/
        JPEGImages            -> symlink to ../JPEGImages   (no copy)
        Annotations/<id>/<id>.jpg   white = Instrument
        ImageSets/val.txt           213 frames
    eval_tissue/
        JPEGImages            -> symlink to ../JPEGImages
        Annotations/<id>/<id>.jpg   white = Soft Tissue
        ImageSets/val.txt           209 frames

To evaluate:  set test_data_path -> eval_instrument (or eval_tissue),
              split -> ImageSets/val.txt,  zero_ann: false, frame_num: 1.
"""

import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

DP = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced")
TM = DP / "tissue mask"
CLASSES = [("instrument", "instrument_seg"), ("tissue", "tissue_seg")]

def ids_from_val(val_path: Path):
    ids = []
    for line in open(val_path):
        line = line.strip()
        if line:
            ids.append(line.split()[0].rstrip("/").split("/")[-1])
    return ids

def main():
    for cls, seg in CLASSES:
        base = DP / f"eval_{cls}"
        base.mkdir(parents=True, exist_ok=True)

        # 1) JPEGImages symlink (recreate to be safe)
        jl = base / "JPEGImages"
        if jl.is_symlink() or jl.exists():
            jl.unlink()
        os.symlink(DP / "JPEGImages", jl)

        # 2) ImageSets/val.txt
        (base / "ImageSets").mkdir(parents=True, exist_ok=True)
        shutil.copy(TM / seg / "val.txt", base / "ImageSets" / "val.txt")

        # 3) Annotations/<id>/<id>.jpg  (white = foreground)
        ann_root = base / "Annotations"
        ids = ids_from_val(TM / seg / "val.txt")
        sizes = set()
        for i in ids:
            src = TM / seg / "pre" / f"{i}.png"
            arr = np.array(Image.open(src).convert("L"))      # 0/255
            sizes.add(arr.shape)
            out = ann_root / i / f"{i}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).convert("RGB").save(out, quality=100)

        print(f"[{cls:10s}] frames={len(ids)}  ann_sizes={sizes}  -> {base}")

if __name__ == "__main__":
    main()

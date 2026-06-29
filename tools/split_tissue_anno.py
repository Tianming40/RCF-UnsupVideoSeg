#!/usr/bin/env python3
"""
split_tissue_anno.py

Split the merged colour annotation (CVAT-style SegmentationClass) into two
binary 0/255 masks per frame:

    instrument_seg/<...>.png   white where Instrument  (144,155,48)
    tissue_seg/<...>.png       white where Soft Tissue (102,255,102)

Every annotated frame produces BOTH masks; if a class is absent in a frame its
mask is all-zero.  Sub-directory structure under SegmentationClass is mirrored.

labelmap.txt:
    Instrument  : 144,155,48
    Soft Tissue : 102,255,102
    background  : 0,0,0
"""

import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced/tissue mask")
SRC  = ROOT / "SegmentationClass"
INST_DIR = ROOT / "instrument_seg"
TIS_DIR  = ROOT / "tissue_seg"

INSTRUMENT = (144, 155, 48)
TISSUE     = (102, 255, 102)

def main():
    files = sorted(SRC.rglob("*.png"))
    n_inst = n_tis = n_both = n_empty = 0
    for f in files:
        rel = f.relative_to(SRC)
        arr = np.array(Image.open(f).convert("RGB"))
        inst = np.all(arr == INSTRUMENT, axis=-1)
        tis  = np.all(arr == TISSUE,     axis=-1)

        has_i, has_t = bool(inst.any()), bool(tis.any())
        n_inst  += has_i
        n_tis   += has_t
        n_both  += (has_i and has_t)
        n_empty += (not has_i and not has_t)

        for base, m in ((INST_DIR, inst), (TIS_DIR, tis)):
            out = base / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m.astype(np.uint8) * 255)).save(out)

    print(f"total frames     : {len(files)}")
    print(f"has instrument   : {n_inst}")
    print(f"has tissue       : {n_tis}")
    print(f"has both         : {n_both}")
    print(f"empty (neither)  : {n_empty}")
    print(f"\ninstrument_seg -> {INST_DIR}")
    print(f"tissue_seg     -> {TIS_DIR}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Reorganize raw CMC grasp-0 images into the VideoDataset format.

  SRC_PRE   /media/.../CMC/grasp-0/pre/   →  STEM.png     (601 files)
  SRC_POST  /media/.../CMC/grasp-0/post/  →  STEM_1.png   (601 files)

Output layout  (DST = CMC_grasp0_from_raw):

  JPEGImages/<STEM>/
    <STEM>.png       ← symlink → SRC_PRE/<STEM>.png
    <STEM>_1.png     ← symlink → SRC_POST/<STEM>_1.png
  ImageSets/train.txt, val.txt, trainval.txt
  Flows_NewCT/         ← empty, filled by generate_flows_cmc.py
  BackwardFlows_NewCT/ ← empty, filled by generate_flows_cmc.py

Usage:
  python tools/reorganize_cmc_grasp0.py
"""

import argparse
import random
from pathlib import Path

SRC_PRE  = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/pre")
SRC_POST = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/post")
DST      = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_from_raw")


def build_split_line(base_id: str) -> str:
    return f"JPEGImages/{base_id}/ {base_id}.png {base_id}_1.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pre_files = sorted(SRC_PRE.glob("*.png"))
    print(f"Found {len(pre_files)} files in pre/")

    valid_pairs = []
    missing_post = []
    for pre_path in pre_files:
        stem = pre_path.stem
        post_path = SRC_POST / f"{stem}_1.png"
        if post_path.exists():
            valid_pairs.append(stem)
        else:
            missing_post.append(stem)

    print(f"Valid pairs: {len(valid_pairs)}")
    if missing_post:
        print(f"  [warn] {len(missing_post)} pre frames have no matching post frame")

    random.seed(args.seed)
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)
    n_val     = max(1, int(len(shuffled) * args.val_ratio))
    val_ids   = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    print(f"  train: {len(train_ids)}   val: {len(val_ids)}")

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "ImageSets").mkdir(exist_ok=True)
    (DST / "Flows_NewCT").mkdir(exist_ok=True)
    (DST / "BackwardFlows_NewCT").mkdir(exist_ok=True)

    created = skipped = 0
    for stem in valid_pairs:
        seq_dir = DST / "JPEGImages" / stem
        seq_dir.mkdir(parents=True, exist_ok=True)
        for src, dst_link in [
            (SRC_PRE  / f"{stem}.png",   seq_dir / f"{stem}.png"),
            (SRC_POST / f"{stem}_1.png", seq_dir / f"{stem}_1.png"),
        ]:
            if dst_link.exists() or dst_link.is_symlink():
                skipped += 1
            else:
                dst_link.symlink_to(src.resolve())
                created += 1

    print(f"  Symlinks created: {created}   already existed: {skipped}")

    def write_txt(path, ids):
        with open(path, "w") as f:
            f.write("\n".join(build_split_line(i) for i in ids) + "\n")

    write_txt(DST / "ImageSets" / "train.txt",    train_ids)
    write_txt(DST / "ImageSets" / "val.txt",      val_ids)
    write_txt(DST / "ImageSets" / "trainval.txt", sorted(valid_pairs))

    print(f"\nDone.  Output: {DST}")
    print("Next step: python tools/deinterlace_cmc_grasp0.py")


if __name__ == "__main__":
    main()

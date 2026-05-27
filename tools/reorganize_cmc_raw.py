#!/usr/bin/env python3
"""
Reorganize raw CMC grasp-10 images into the VideoDataset format expected by
the RCF training pipeline, starting directly from the original source:

  SRC_PRE   /media/.../CMC/grasp-10/pre/   →  STEM.png     (601 files)
  SRC_POST  /media/.../CMC/grasp-10/post/  →  STEM_1.png   (601 files)

Output layout  (DST = CMC_grasp10_from_raw):

  JPEGImages/
    <STEM>/
      <STEM>.png       ← symlink → SRC_PRE/<STEM>.png
      <STEM>_1.png     ← symlink → SRC_POST/<STEM>_1.png
  ImageSets/
    train.txt
    val.txt
    trainval.txt
  Flows_NewCT/         ← empty, to be filled by generate_flows_cmc.py
  BackwardFlows_NewCT/ ← empty, to be filled by generate_flows_cmc.py

ImageSets line format (same as CMC_grasp10_finetune_deinterlaced):
  JPEGImages/<STEM>/  <STEM>.png  <STEM>_1.png

No deinterlacing is applied here.  Run check_interlace.py first to decide
whether deinterlacing is needed.

Usage:
  python tools/reorganize_cmc_raw.py
  python tools/reorganize_cmc_raw.py --val_ratio 0.2 --seed 42
"""

import argparse
import os
import random
from pathlib import Path

SRC_PRE  = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-10/pre")
SRC_POST = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-10/post")
DST      = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_from_raw")


def build_split_line(base_id: str) -> str:
    return f"JPEGImages/{base_id}/ {base_id}.png {base_id}_1.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── 1. Collect valid pairs ──────────────────────────────────────────
    pre_files = sorted(SRC_PRE.glob("*.png"))
    print(f"Found {len(pre_files)} files in pre/")

    valid_pairs = []
    missing_post = []
    for pre_path in pre_files:
        stem = pre_path.stem          # e.g. "96391832500300027230"
        post_path = SRC_POST / f"{stem}_1.png"
        if post_path.exists():
            valid_pairs.append(stem)
        else:
            missing_post.append(stem)

    print(f"Valid pairs: {len(valid_pairs)}")
    if missing_post:
        print(f"  [warn] {len(missing_post)} pre frames have no matching post frame:")
        for s in missing_post[:10]:
            print(f"    {s}")

    # ── 2. Train / val split ────────────────────────────────────────────
    random.seed(args.seed)
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)
    n_val     = max(1, int(len(shuffled) * args.val_ratio))
    val_ids   = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    print(f"  train: {len(train_ids)}   val: {len(val_ids)}")

    # ── 3. Create directory skeleton ────────────────────────────────────
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "ImageSets").mkdir(exist_ok=True)
    (DST / "Flows_NewCT").mkdir(exist_ok=True)
    (DST / "BackwardFlows_NewCT").mkdir(exist_ok=True)

    # ── 4. Per-pair JPEGImages subdirs with symlinks ─────────────────────
    created = 0
    skipped = 0
    for stem in valid_pairs:
        seq_dir = DST / "JPEGImages" / stem
        seq_dir.mkdir(parents=True, exist_ok=True)

        links = [
            (SRC_PRE  / f"{stem}.png",    seq_dir / f"{stem}.png"),
            (SRC_POST / f"{stem}_1.png",  seq_dir / f"{stem}_1.png"),
        ]
        for src, dst_link in links:
            if dst_link.exists() or dst_link.is_symlink():
                skipped += 1
            else:
                dst_link.symlink_to(src.resolve())
                created += 1

    print(f"  Symlinks created: {created}   already existed: {skipped}")

    # ── 5. Write ImageSets txt files ─────────────────────────────────────
    def write_txt(path: Path, ids: list):
        with open(path, "w") as f:
            f.write("\n".join(build_split_line(i) for i in ids) + "\n")

    write_txt(DST / "ImageSets" / "train.txt",    train_ids)
    write_txt(DST / "ImageSets" / "val.txt",      val_ids)
    write_txt(DST / "ImageSets" / "trainval.txt", sorted(valid_pairs))

    print(f"\nDone.  Output: {DST}")
    print(f"  ImageSets/train.txt    ({len(train_ids)} pairs)")
    print(f"  ImageSets/val.txt      ({len(val_ids)} pairs)")
    print(f"  ImageSets/trainval.txt ({len(valid_pairs)} pairs)")
    print()
    print("Next steps:")
    print("  1. (Optional) python tools/check_interlace.py  →  decide if deinterlacing is needed")
    print("  2. python RAFT/generate_flows_cmc.py           →  fill Flows_NewCT/")
    print("  3. Update configs/instrument/rcf_cmc_finetune*.yaml:")
    print(f"       data_path: {DST}")


if __name__ == "__main__":
    main()

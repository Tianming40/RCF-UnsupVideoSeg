#!/usr/bin/env python3
"""


Usage:
  python tools/make_vis_video.py \
    --vis_dir  analysis/dino_val_vis/vis \
    --output   analysis/dino_val_vis \
    --fps      10
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vis_dir", required=True, help="tissue_vis 生成的图像目录")
    p.add_argument("--output",  required=True, help="视频保存目录")
    p.add_argument("--fps",     type=float, default=10)
    args = p.parse_args()

    vis_dir = Path(args.vis_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

   
    groups = defaultdict(list)
    pattern = re.compile(r"^(.+)_(\d+)\.jpg$")

    for f in sorted(vis_dir.glob("*.jpg")):
        m = pattern.match(f.name)
        if m:
            seq, frame = m.group(1), int(m.group(2))
            groups[seq].append((frame, f))

    if not groups:
        print(f"No .jpg files found in {vis_dir}")
        return

    for seq, frames in sorted(groups.items()):
        frames.sort(key=lambda x: x[0])
        print(f"\n{seq}: {len(frames)} frames")

        # size
        first = cv2.imread(str(frames[0][1]))
        if first is None:
            print(f"  Cannot read {frames[0][1]}, skip")
            continue
        h, w = first.shape[:2]

        out_path = out_dir / f"{seq}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )

        for i, (fnum, fpath) in enumerate(frames):
            img = cv2.imread(str(fpath))
            if img is None:
                print(f"  Skip unreadable frame {fpath.name}")
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)

        writer.release()
        print(f"  → {out_path}  ({w}x{h} @ {args.fps}fps)")

    print("\nDone.")


if __name__ == "__main__":
    main()

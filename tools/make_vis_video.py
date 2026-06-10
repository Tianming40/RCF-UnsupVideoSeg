#!/usr/bin/env python3
"""
Assemble visualization jpg images into mp4 video(s).

Two modes:
  default   : one mp4 per sequence
  --single  : all sequences concatenated into one mp4 (e.g. 601 clips x 2 frames)

Filename formats supported:
  tissue_vis_inference : {seq_name}_{frame_id}.jpg       (frame_id is a short integer)
  grasp10_gif_vis      : {seq_name}_{seq_name}.jpg       (frame 0)
                         {seq_name}_{seq_name}_1.jpg     (frame 1)

Usage:
  # One video per sequence
  python tools/make_vis_video.py \
    --vis_dir  saved/grasp10_gif_vis/gif \
    --output   saved/grasp10_gif_vis \
    --fps      3

  # All sequences concatenated into one video
  python tools/make_vis_video.py \
    --vis_dir  saved/grasp10_gif_vis/gif \
    --output   saved/grasp10_gif_vis \
    --fps      3 \
    --single
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2


def parse_groups(vis_dir: Path):
    """
    Returns {seq_name: [(sort_key, Path), ...]} parsed from filenames.

    Supports:
      {seq}_{frame_digits}.jpg      -> sort_key = frame_digits (int)
      {seq}_{seq}.jpg               -> sort_key = 0
      {seq}_{seq}_1.jpg             -> sort_key = 1
    """
    groups = defaultdict(list)
    # tissue_vis style: frame id is a short integer (1-6 digits)
    pat_tissue = re.compile(r"^(.+)_(\d{1,6})\.jpg$")
    # grasp_gif_vis style: seq is a long alphanumeric ID, frame name matches seq (optionally + _1)
    pat_grasp  = re.compile(r"^([A-Za-z0-9]+)_[A-Za-z0-9]+(_1)?\.jpg$")

    for f in sorted(vis_dir.glob("*.jpg")):
        name = f.name
        mg = pat_grasp.match(name)
        if mg:
            seq       = mg.group(1)
            is_frame1 = mg.group(2) is not None
            groups[seq].append((1 if is_frame1 else 0, f))
            continue
        mt = pat_tissue.match(name)
        if mt:
            seq, fnum = mt.group(1), int(mt.group(2))
            groups[seq].append((fnum, f))

    return groups


def write_video(out_path: Path, frames: list, fps: float, ref_size=None):
    """frames: [(sort_key, Path), ...], already sorted."""
    first = cv2.imread(str(frames[0][1]))
    if first is None:
        print(f"  Cannot read {frames[0][1]}, skip")
        return None
    h, w = first.shape[:2] if ref_size is None else ref_size
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for _, fpath in frames:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return (w, h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vis_dir", required=True, help="Directory containing jpg images")
    p.add_argument("--output",  required=True, help="Output directory for video(s)")
    p.add_argument("--fps",     type=float, default=3)
    p.add_argument("--single",  action="store_true",
                   help="Concatenate all sequences into one video")
    args = p.parse_args()

    vis_dir = Path(args.vis_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = parse_groups(vis_dir)
    if not groups:
        print(f"No .jpg files found in {vis_dir}")
        return

    print(f"Found {len(groups)} sequences, {sum(len(v) for v in groups.values())} frames total")

    if args.single:
        out_path    = out_dir / "all_sequences.mp4"
        first_seq   = sorted(groups.keys())[0]
        first_frame = sorted(groups[first_seq])[0][1]
        ref_img     = cv2.imread(str(first_frame))
        if ref_img is None:
            print("Cannot read first frame, abort.")
            return
        h, w = ref_img.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter.fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )
        total = 0
        for seq in sorted(groups.keys()):
            frames = sorted(groups[seq], key=lambda x: x[0])
            for _, fpath in frames:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                writer.write(img)
                total += 1
        writer.release()
        print(f"-> {out_path}  ({w}x{h} @ {args.fps}fps, {total} frames, {len(groups)} sequences)")
    else:
        for seq, frames in sorted(groups.items()):
            frames.sort(key=lambda x: x[0])
            out_path = out_dir / f"{seq}.mp4"
            size = write_video(out_path, frames, args.fps)
            if size:
                print(f"  {seq}: {len(frames)} frames -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

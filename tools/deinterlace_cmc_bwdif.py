#!/usr/bin/env python3
"""
Adaptive motion-aware deinterlacing for CMC frames (grasp0/5/10), replacing
the naive row-duplication used to build CMC_grasp0_5_10_merged.

Old approach (deinterlace_cmc_grasp0.py): out[0::2] = out[1::2] — every
even row is HARD-REPLACED with an exact copy of the row below it,
unconditionally, on every frame. Verified: 100%-exact pixel duplication on
every even row, destroying half the image's real vertical information even
in frames that were never interlaced to begin with.

This script:
  1. Detects interlacing PER SEQUENCE using ffmpeg's `idet` filter, fed the
     sequence's own real (pre, post) frame pair concatenated into a genuine
     2-frame stream — NOT a duplicated/looped single frame (idet, like
     bwdif, needs authentic inter-frame difference to classify reliably; a
     static self-loop gives it nothing to detect and returns Undetermined
     for everything). Full-dataset scan (601 sequences x 3 grasp offsets,
     3606 frames) found ~73% TFF-interlaced, ~26.5% genuinely Progressive
     (not interlaced at all — same ~318 sequences across g0/g5/g10,
     consistent with those cases sharing recording equipment/settings) —
     so a blanket "deinterlace everything" pass would be wrong: it would
     needlessly smooth ~1/4 of the data that was never interlaced.
  2. Only sequences classified TFF/BFF get bwdif applied — using bwdif's
     motion-adaptive bob-weave algorithm (static regions keep real
     information from both fields; only true motion regions fall back to
     spatial interpolation), fed the REAL (pre, post) pair (not a
     duplicated frame) so its temporal motion estimation has genuine
     signal to work with. Progressive/Undetermined sequences are copied
     through unchanged.

Source: CMC/grasp-{0,5,10}/{pre,post}/ (the true raw source; the existing
CMC_grasp{0,5,10}_from_raw/ dirs are just symlinks into this, reused here
for convenience).

Output: mimics CMC_grasp0_5_10_merged/'s directory STRUCTURE exactly
(JPEGImages/<stem>_g{0,5,10}/<stem>_g{0,5,10}.png + _1.png) but with
entirely new CONTENT (adaptively deinterlaced from raw, not the old
duplication). Written to CMC_grasp0_5_10_merged_bwdif/ — a NEW dataset
root, does NOT touch/overwrite CMC_grasp0_5_10_merged/ that current
trained models and precomputed RAFT flows depend on. ImageSets/*.txt are
copied verbatim from the existing merged dataset (same case IDs / split
membership — only pixel content changes, so the split files stay valid).

RAFT flows must be regenerated on the new images before they can be used
for training (see RAFT/generate_flows_cmc.py) — Flows_NewCT/
BackwardFlows_NewCT are NOT produced by this script.

Usage:
  # 1. Visual QC on a handful of sequences first:
  python tools/deinterlace_cmc_bwdif.py --test-only --n-test 6

  # 2. Process everything (all three grasp offsets):
  python tools/deinterlace_cmc_bwdif.py
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

GRASP_OFFSETS = ["0", "5", "10"]
DATA_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset")
RAW_ROOT = DATA_ROOT / "CMC"
MERGED_SRC = DATA_ROOT / "CMC_grasp0_5_10_merged"          # for ImageSets only
OUT_ROOT = DATA_ROOT / "CMC_grasp0_5_10_merged_bwdif"


def detect_parity(pre: Path, post: Path) -> str:
    """Returns 'tff', 'bff', 'progressive', or 'undetermined' — majority
    vote of idet's multi-frame classification over the real (pre, post) pair."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "info",
        "-i", str(pre), "-i", str(post),
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,idet[out]",
        "-map", "[out]", "-frames:v", "2", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stderr
    lines = [l for l in text.splitlines() if "Multi frame detection" in l]
    if not lines:
        return "undetermined"
    m = re.search(r"TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)", lines[-1])
    if not m:
        return "undetermined"
    tff, bff, prog, undet = map(int, m.groups())
    counts = {"tff": tff, "bff": bff, "progressive": prog, "undetermined": undet}
    return max(counts, key=counts.get)


def deinterlace_pair(pre: Path, post: Path, dst_pre: Path, dst_post: Path, parity: int):
    """bwdif on the genuine (pre, post) 2-frame stream — both frames get
    authentic temporal context (no duplicated-frame workaround needed)."""
    dst_pre.parent.mkdir(parents=True, exist_ok=True)
    tmp_pattern = str(dst_pre.parent / f".{dst_pre.stem}_bwdif_tmp_%d.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(pre), "-i", str(post),
        "-filter_complex", f"[0:v][1:v]concat=n=2:v=1:a=0,bwdif=mode=0:parity={parity}:deint=0[out]",
        "-map", "[out]", "-frames:v", "2",
        tmp_pattern,
    ]
    subprocess.run(cmd, check=True)
    Path(str(dst_pre.parent / f".{dst_pre.stem}_bwdif_tmp_1.png")).replace(dst_pre)
    Path(str(dst_pre.parent / f".{dst_pre.stem}_bwdif_tmp_2.png")).replace(dst_post)


def process_sequence(pre: Path, post: Path, dst_pre: Path, dst_post: Path):
    label = detect_parity(pre, post)
    if label == "tff":
        deinterlace_pair(pre, post, dst_pre, dst_post, parity=0)
    elif label == "bff":
        deinterlace_pair(pre, post, dst_pre, dst_post, parity=1)
    else:  # progressive or undetermined -> leave unchanged
        dst_pre.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pre, dst_pre)
        shutil.copy2(post, dst_post)
    return label


def copy_imagesets():
    src_imagesets = MERGED_SRC / "ImageSets"
    dst_imagesets = OUT_ROOT / "ImageSets"
    if not src_imagesets.exists():
        print(f"WARNING: {src_imagesets} not found, skipping ImageSets copy")
        return
    dst_imagesets.mkdir(exist_ok=True, parents=True)
    for txt in src_imagesets.glob("*.txt"):
        dst_txt = dst_imagesets / txt.name
        if not dst_txt.exists():
            shutil.copy2(txt, dst_txt)
            print(f"  Copied ImageSets/{txt.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-only", action="store_true",
                     help="process only a handful of sequences into a scratch dir for visual QC; "
                          "does not touch real dataset output dirs")
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--test-out",
                     default="/tmp/claude-1011/-media-mitiadmin-Micron-7450-1-tianming-RCF-UnsupVideoSeg/"
                             "967cb7c3-a9f6-4c1e-b12e-d38159ada2cc/scratchpad/interlace_check/bwdif_merged_test")
    args = ap.parse_args()

    if args.test_only:
        src_root = RAW_ROOT / "grasp-0"
        stems = sorted(p.stem for p in (src_root / "pre").glob("*.png"))[:args.n_test]
        out_dir = Path(args.test_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            pre = src_root / "pre" / f"{stem}.png"
            post = src_root / "post" / f"{stem}_1.png"
            if not (pre.exists() and post.exists()):
                print(f"  SKIP {stem}: missing pre/post")
                continue
            out_stem = f"{stem}_g0"
            dst_pre = out_dir / out_stem / f"{out_stem}.png"
            dst_post = out_dir / out_stem / f"{out_stem}_1.png"
            label = process_sequence(pre, post, dst_pre, dst_post)
            print(f"  {out_stem}: {label} -> {dst_pre.parent}")
        print(f"\nTest output: {out_dir}")
        print("Inspect visually before running the full batch.")
        return

    label_counts = {"tff": 0, "bff": 0, "progressive": 0, "undetermined": 0}
    for g in GRASP_OFFSETS:
        src_root = RAW_ROOT / f"grasp-{g}"
        pre_dir = src_root / "pre"
        post_dir = src_root / "post"
        if not (pre_dir.exists() and post_dir.exists()):
            print(f"SKIP grasp{g}: source not found at {src_root}")
            continue
        stems = sorted(p.stem for p in pre_dir.glob("*.png"))
        print(f"grasp{g}: {len(stems)} sequences")
        for stem in tqdm(stems, desc=f"grasp{g}"):
            pre = pre_dir / f"{stem}.png"
            post = post_dir / f"{stem}_1.png"
            if not (pre.exists() and post.exists()):
                continue
            out_stem = f"{stem}_g{g}"
            dst_pre = OUT_ROOT / "JPEGImages" / out_stem / f"{out_stem}.png"
            dst_post = OUT_ROOT / "JPEGImages" / out_stem / f"{out_stem}_1.png"
            if dst_pre.exists() and dst_post.exists():
                continue
            label = process_sequence(pre, post, dst_pre, dst_post)
            label_counts[label] += 1

    copy_imagesets()

    print(f"\nDone. {label_counts}")
    print(f"Output: {OUT_ROOT}")
    print("Next step: regenerate RAFT flows on the new deinterlaced images before training on them.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inspect CMC images to determine:
  1. Whether they are actually interlaced (comb artifacts in motion regions)
  2. Which field (even/odd rows) is the "current" frame
  3. Which deinterlace strategy to use

Output:
  tools/interlace_check/   – side-by-side crops of even/odd fields + difference maps

Usage:
  python tools/check_interlace.py
  python tools/check_interlace.py --pair 96391832500300027230  # specific pair stem
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

JPEG_ROOT = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_from_raw/JPEGImages")
OUT = Path("tools/interlace_check_raw")


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def extract_fields(img):
    """Return (even_field, odd_field) each as full-height images (rows duplicated)."""
    even = img.copy(); even[1::2] = even[0::2]   # keep even, fill odd from even above
    odd  = img.copy(); odd[0::2]  = odd[1::2]    # keep odd,  fill even from odd below
    return even, odd


def row_diff_map(img):
    """Per-pixel absolute difference between consecutive rows → indicates interlacing."""
    diff = np.abs(img[1:].astype(np.int16) - img[:-1].astype(np.int16)).mean(axis=2)
    return diff.astype(np.uint8)


def analyse_pair(pre_path, post_path, out_dir: Path, label: str):
    """
    pre  = frame N   (e.g. 96391832500300027230.png)
    post = frame N+1 (e.g. 96391832500300027230_1.png)

    If the images are interlaced in the typical video sense:
      - post contains: even rows from time T,  odd rows from time T-1
      OR
      - post contains: odd rows from time T,   even rows from time T-1

    We check which field of POST looks more like PRE (=T-1) to figure out
    which rows are the "old" field.
    """
    pre  = load(pre_path)
    post = load(post_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Row-difference maps ──────────────────────────────────────────
    diff_pre  = row_diff_map(pre)
    diff_post = row_diff_map(post)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(diff_pre,  cmap='hot', vmax=40); axes[0].set_title("Row-diff PRE (frame N)")
    axes[1].imshow(diff_post, cmap='hot', vmax=40); axes[1].set_title("Row-diff POST (frame N+1)")
    fig.suptitle(f"{label} – Row-difference maps\n"
                 "High alternating-row diff = interlacing artifact")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_row_diff.png", dpi=120)
    plt.close(fig)

    # ── 2. Even vs Odd field of POST ────────────────────────────────────
    post_even, post_odd = extract_fields(post)

    # How similar is each field of POST to PRE?
    mae_even_vs_pre = np.abs(post_even.astype(np.int16) - pre.astype(np.int16)).mean()
    mae_odd_vs_pre  = np.abs(post_odd.astype(np.int16)  - pre.astype(np.int16)).mean()

    print(f"\n[{label}]")
    print(f"  MAE(POST-even-field  vs PRE) = {mae_even_vs_pre:.2f}")
    print(f"  MAE(POST-odd-field   vs PRE) = {mae_odd_vs_pre:.2f}")
    if mae_even_vs_pre < mae_odd_vs_pre:
        print("  → EVEN rows of POST ≈ PRE  ⇒  EVEN rows are OLD field (from T-1)")
        print("     Correct deinterlace: KEEP ODD rows (current time), duplicate to even")
    else:
        print("  → ODD rows of POST ≈ PRE   ⇒  ODD rows are OLD field (from T-1)")
        print("     Correct deinterlace: KEEP EVEN rows (current time), duplicate to odd  ← current code")

    # ── 3. Side-by-side crop visualisation ─────────────────────────────
    # Pick a 200×200 crop from the centre
    h, w = post.shape[:2]
    cy, cx = h // 2, w // 2
    s = 100
    crop = lambda img: img[cy-s:cy+s, cx-s:cx+s]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(
        axes,
        [crop(pre), crop(post), crop(post_even), crop(post_odd)],
        ["PRE (frame N)", "POST raw (N+1)", "POST even-field only", "POST odd-field only"]
    ):
        ax.imshow(img); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{label} – centre crop  |  MAE even={mae_even_vs_pre:.1f}  odd={mae_odd_vs_pre:.1f}\n"
                 "Look for horizontal comb artifacts in 'POST raw'")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_fields.png", dpi=120)
    plt.close(fig)

    # ── 4. Alternating-row mean difference (scalar: high = interlaced) ──
    # Compare row i vs row i+2 (same parity) – should be small
    # Compare row i vs row i+1 (diff parity) – large if interlaced
    same_parity = np.abs(post[2:].astype(np.int16) - post[:-2].astype(np.int16)).mean()
    diff_parity = np.abs(post[1:].astype(np.int16) - post[:-1].astype(np.int16)).mean()
    interlace_ratio = diff_parity / (same_parity + 1e-6)
    print(f"  Same-parity row diff: {same_parity:.2f}   "
          f"Adjacent row diff: {diff_parity:.2f}   "
          f"Ratio: {interlace_ratio:.2f}  (>2 suggests interlacing)")

    return mae_even_vs_pre, mae_odd_vs_pre, interlace_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None,
                        help="Stem of the pair to inspect (without _1.png suffix). "
                             "Default: pick 5 pairs automatically.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of random pairs to sample (if --pair not given).")
    args = parser.parse_args()

    # Collect all (pre, post) pairs from subdirectories
    all_post = sorted(JPEG_ROOT.glob("*/*_1.png"))
    if not all_post:
        raise FileNotFoundError(f"No *_1.png files found under {JPEG_ROOT}")

    if args.pair:
        seq_dir = JPEG_ROOT / args.pair
        pairs = [(seq_dir / f"{args.pair}.png", seq_dir / f"{args.pair}_1.png")]
    else:
        # Each subdir is one pair: JPEG_ROOT/<STEM>/<STEM>.png + <STEM>_1.png
        all_stems = sorted([d.name for d in JPEG_ROOT.iterdir() if d.is_dir()])
        step = max(1, len(all_stems) // args.n)
        sampled = all_stems[::step][:args.n]
        pairs = [
            (JPEG_ROOT / stem / f"{stem}.png",
             JPEG_ROOT / stem / f"{stem}_1.png")
            for stem in sampled
        ]

    print(f"Checking {len(pairs)} pairs → output in {OUT}/")

    even_wins = 0; odd_wins = 0; interlace_ratios = []
    for pre_path, post_path in pairs:
        if not pre_path.exists():
            print(f"  [skip] PRE not found: {pre_path}")
            continue
        label = post_path.stem  # e.g. 96391832500300027230_1
        mae_even, mae_odd, ratio = analyse_pair(pre_path, post_path, OUT, label)
        interlace_ratios.append(ratio)
        if mae_even < mae_odd:
            even_wins += 1
        else:
            odd_wins += 1

    print("\n══════════════════════════════════════════")
    print(f"  Even-field ≈ PRE (old field):  {even_wins} / {len(interlace_ratios)} pairs")
    print(f"  Odd-field  ≈ PRE (old field):  {odd_wins} / {len(interlace_ratios)} pairs")
    print(f"  Mean interlace ratio: {np.mean(interlace_ratios):.2f}  (>2 = likely interlaced)")
    print()
    if np.mean(interlace_ratios) < 1.5:
        print("  ✗ Images do NOT appear to be interlaced – no deinterlacing needed!")
    elif odd_wins > even_wins:
        print("  ✓ ODD rows are the OLD field → KEEP EVEN rows (current code is CORRECT)")
    else:
        print("  ✗ EVEN rows are the OLD field → should KEEP ODD rows instead!")
        print("    Fix in deinterlace_cmc.py:  out[0::2] = out[1::2]  (not out[1::2] = out[0::2])")
    print("══════════════════════════════════════════")
    print(f"\nVisuals saved to: {OUT}/")


if __name__ == "__main__":
    main()

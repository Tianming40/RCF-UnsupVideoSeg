#!/usr/bin/env python3
"""
grasp0_annotation_vis.py

For the 601 grasp0 frame-pairs:
  1. Parse grasping_points/grasp_dissect_annotations/ JSON files.
       - JSON named <seq>.json   → real annotation
       - JSON named _<seq>.json  → explicitly no-grasp (ignored per spec)
       - No JSON at all          → missing
  2. Re-generate annotation_summary.txt (same format as the existing one).
  3. For every annotated sequence write a 2-frame GIF to gif_annotated/:
       - Both frames are shown side by side in time.
       - Dissection point → lime-green  cross + circle
       - Grasp point      → cyan-blue   cross + circle
       - Multiple annotations → draw all of them.

Usage:
    python tools/grasp0_annotation_vis.py \\
        --data  /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced \\
        --split ImageSets/trainval_single.txt \\
        --ann_dir grasping_points/grasp_dissect_annotations \\
        --out_gif gif_annotated \\
        --out_summary annotation_summary_new.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ── Colours (match gif2 reference exactly) ────────────────────────────────────
LIME  = (57, 255, 20)    # dissection point  D  → palette index 254
CYAN  = (0, 200, 255)    # grasp point       G  → palette index 255
MARKER_R   = 14
CROSS_HALF = 20


# ── Draw marker directly onto palette-mode image ──────────────────────────────

def _draw_marker_p(px, cx: int, cy: int, idx: int, W: int, H: int):
    r, cl = MARKER_R, CROSS_HALF
    for dy in range(-r - 1, r + 2):
        for dx in range(-r - 1, r + 2):
            if abs((dx*dx + dy*dy)**0.5 - r) < 1.5:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    px[nx, ny] = idx
    for d in range(-cl, cl + 1):
        for nx, ny in [(cx + d, cy), (cx, cy + d)]:
            if 0 <= nx < W and 0 <= ny < H:
                px[nx, ny] = idx


# ── Annotation loader ──────────────────────────────────────────────────────────

def load_ann_index(ann_dir: Path) -> dict:
    """
    Returns {seq_name: ('valid', [ann, ...]) | ('no_grasp', []) | ('missing', [])}
    where each ann = (frame_key_str, [D_x, D_y], [G_x, G_y])
    """
    index = {}
    for f in ann_dir.iterdir():
        if not f.suffix == '.json':
            continue
        stem = f.stem
        if stem.startswith('_'):
            seq = stem[1:]
            index[seq] = ('no_grasp', [])
        else:
            try:
                d = json.loads(f.read_text())
                anns = []
                for ann_dict in d.get('annotations', []):
                    for frame_key, pts in ann_dict.items():
                        if len(pts) >= 2:
                            anns.append((frame_key, pts[0], pts[1]))
                if anns:
                    index[stem] = ('valid', anns)
                else:
                    index[stem] = ('no_grasp', [])
            except Exception:
                index[stem] = ('no_grasp', [])
    return index


# ── GIF creation ───────────────────────────────────────────────────────────────

def _to_gif_frame(img_rgb: Image.Image, annotations: list) -> Image.Image:
    W, H = img_rgb.size
    # Quantize to 254 colours, reserve last 2 slots for LIME/CYAN
    img_p = img_rgb.quantize(colors=254, dither=Image.Dither.NONE)
    pal = img_p.getpalette()
    pal[254*3:254*3+3] = list(LIME)
    pal[255*3:255*3+3] = list(CYAN)
    img_p.putpalette(pal)
    px = img_p.load()
    for _, d_xy, g_xy in annotations:
        _draw_marker_p(px, int(d_xy[0]*W), int(d_xy[1]*H), 254, W, H)
        _draw_marker_p(px, int(g_xy[0]*W), int(g_xy[1]*H), 255, W, H)
    return img_p


def make_annotated_gif(frame0_path: Path, frame1_path: Path,
                       annotations: list, out_path: Path):
    img0 = Image.open(frame0_path).convert('RGB')
    img1 = Image.open(frame1_path).convert('RGB')
    f0 = _to_gif_frame(img0, annotations)
    f1 = _to_gif_frame(img1, annotations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f0.save(str(out_path), save_all=True, append_images=[f1],
            duration=500, loop=0, format='GIF')


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(seqs: list, ann_index: dict, out_path: Path):
    valid   = [(s, ann_index[s][1]) for s in seqs if ann_index.get(s, ('missing',))[0] == 'valid']
    no_grasp = [s for s in seqs if ann_index.get(s, ('missing',))[0] == 'no_grasp']
    missing  = [s for s in seqs if s not in ann_index]

    by_count = defaultdict(list)
    for s, anns in valid:
        by_count[len(anns)].append((s, anns))

    lines = []
    SEP = '=' * 80
    lines += [
        SEP,
        'CMC_grasp0_deinterlaced — Annotation Summary',
        SEP,
        '',
        'OVERVIEW',
        '-' * 40,
        f'Total sequences (trainval):        {len(seqs):>4}',
        f'  With valid annotation:            {len(valid):>4}',
    ]
    for cnt in sorted(by_count):
        lines.append(f'    {cnt} annotation(s):              {len(by_count[cnt]):>4}')
    lines += [
        f'  Explicitly no-grasp (underscore): {len(no_grasp):>4}',
        f'  No JSON at all:                   {len(missing):>4}',
        f'  Total without annotation:         {len(no_grasp)+len(missing):>4}',
        '',
    ]

    for cnt in sorted(by_count):
        tag = {1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE'}.get(cnt, f'{cnt}x')
        lines += [
            SEP,
            f'[{cnt}] {tag} ANNOTATION ({len(by_count[cnt])} sequences)',
            f'    {"seq_id":<24}  frame_id  D(x,y)  G(x,y)',
            '-' * 80,
        ]
        for s, anns in sorted(by_count[cnt], key=lambda x: x[0]):
            for fk, d_xy, g_xy in anns:
                lines.append(
                    f'{s}  frame={int(fk):>6}  '
                    f'D=({d_xy[0]:.4f},{d_xy[1]:.4f})  '
                    f'G=({g_xy[0]:.4f},{g_xy[1]:.4f})'
                )
        lines.append('')

    if no_grasp:
        lines += [SEP, f'EXPLICITLY NO-GRASP ({len(no_grasp)} sequences)', '-' * 40]
        lines += sorted(no_grasp)
        lines.append('')

    if missing:
        lines += [SEP, f'NO JSON AT ALL ({len(missing)} sequences)', '-' * 40]
        lines += sorted(missing)
        lines.append('')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    print(f'Summary written → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
    p.add_argument('--split',       default='ImageSets/trainval_single.txt')
    p.add_argument('--ann_dir',     default='grasping_points/grasp_dissect_annotations')
    p.add_argument('--out_gif',     default='analysis/gif_annotated', help='output GIF subdir')
    p.add_argument('--out_summary', default='analysis/annotation_summary.txt')
    p.add_argument('--no_gif',      action='store_true', help='skip GIF generation')
    args = p.parse_args()

    root    = Path(args.data)
    ann_dir = root / args.ann_dir
    gif_dir = root / args.out_gif
    summary_path = root / args.out_summary

    # ── Load 601 sequence names from split file ──────────────────────────────
    seqs = []
    for line in (root / args.split).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        # seq dir is parts[0], e.g. "JPEGImages/96391832500300027230/"
        seqs.append(Path(parts[0]).name)

    print(f'Sequences: {len(seqs)}')

    # ── Build annotation index from JSON files ────────────────────────────────
    ann_index = load_ann_index(ann_dir)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_valid   = sum(1 for s in seqs if ann_index.get(s, ('missing',))[0] == 'valid')
    n_nogrsp  = sum(1 for s in seqs if ann_index.get(s, ('missing',))[0] == 'no_grasp')
    n_missing = sum(1 for s in seqs if s not in ann_index)
    print(f'  valid={n_valid}  no_grasp={n_nogrsp}  missing={n_missing}')

    # ── Write summary ─────────────────────────────────────────────────────────
    write_summary(seqs, ann_index, summary_path)

    # ── Generate GIFs ─────────────────────────────────────────────────────────
    if args.no_gif:
        return

    gif_dir.mkdir(parents=True, exist_ok=True)
    done = skip = 0

    for seq in seqs:
        status, anns = ann_index.get(seq, ('missing', []))

        frame0 = root / 'JPEGImages' / seq / f'{seq}.png'
        frame1 = root / 'JPEGImages' / seq / f'{seq}_1.png'
        out    = gif_dir / f'{seq}.gif'

        if not frame0.exists() or not frame1.exists():
            print(f'  [SKIP] missing frames: {seq}')
            skip += 1
            continue

        if status == 'valid':
            make_annotated_gif(frame0, frame1, anns, out)
        else:
            # no annotation — still make a plain GIF for completeness
            img0 = Image.open(frame0).convert('RGB')
            img1 = Image.open(frame1).convert('RGB')
            out.parent.mkdir(parents=True, exist_ok=True)
            img0.save(str(out), save_all=True, append_images=[img1],
                      duration=500, loop=0, format='GIF')
        done += 1

    print(f'GIFs written → {gif_dir}  ({done} done, {skip} skipped)')


if __name__ == '__main__':
    main()

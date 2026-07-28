"""
Build gap2/gap3 variants of val_paired.txt (see build_paired_eval_split.py's
gap1 version) for CMC_grasp0_continuous_bwdif/eval_instrument and
eval_tissue -- lets v121's joint-mask model be evaluated with a
farther-apart auxiliary frame (frame1 = <stem>_2.png or <stem>_3.png)
instead of only the gap1 neighbour it was built/evaluated with originally.

Usage:
  python tools/build_paired_eval_split_gapN.py --gap 2
  python tools/build_paired_eval_split_gapN.py --gap 3
"""
import argparse
import os
from pathlib import Path

CONTINUOUS_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
SOURCES = ['eval_instrument', 'eval_tissue']


def symlink_force(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap', type=int, required=True)
    args = parser.parse_args()
    gap = args.gap

    for name in SOURCES:
        eval_dir = CONTINUOUS_ROOT / name
        old_split = eval_dir / 'ImageSets' / 'val.txt'
        lines = [l.strip() for l in old_split.read_text().splitlines() if l.strip()]

        new_lines = []
        n_missing = 0
        for line in lines:
            jpeg_dir, frame0 = line.split()
            stem = jpeg_dir.rstrip('/').split('/')[1]
            src_frameN = CONTINUOUS_ROOT / stem / f'{stem}_{gap}.png'
            if not src_frameN.exists():
                n_missing += 1
                continue
            dst_frameN = eval_dir / jpeg_dir.rstrip('/') / f'{stem}_{gap}.png'
            symlink_force(src_frameN, dst_frameN)
            new_lines.append(f'{jpeg_dir} {frame0} {stem}_{gap}.png')

        out_split = eval_dir / 'ImageSets' / f'val_paired_gap{gap}.txt'
        out_split.write_text('\n'.join(new_lines) + '\n')
        print(f'{name} gap{gap}: {len(new_lines)}/{len(lines)} kept ({n_missing} missing frame{gap}) -> {out_split}')


if __name__ == '__main__':
    main()

"""
Build a 2-frame-per-line eval split (val_paired.txt) for
CMC_grasp0_continuous_bwdif/eval_instrument and eval_tissue, needed by
RCFJointMaskSoftTissueModel (models/rcf_joint_mask_model.py) -- its mask
branch needs an auxiliary neighbour frame at eval time too, not just train
time. Frame 0 stays the existing annotated frame (<stem>.png); frame 1 is
the gap1 neighbour (<stem>_1.png, matching v121's mostly-gap1 training
mix), symlinked in from CMC_grasp0_continuous_bwdif/<stem>/ (same source
tools/build_multigap_matched_eval.py already used for frame 0).

Only frame 0 is ever scored/annotated (dataset/data.py's VideoDataset,
relaxed this session to allow frame_num>1 in eval: annotation always
belongs to current_seq[frame_ind], regardless of frame_num) -- frame 1 is
purely auxiliary input for the joint-mask model's forward pass.

Usage:
  python tools/build_paired_eval_split.py
"""
import os
from pathlib import Path

CONTINUOUS_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
SOURCES = ['eval_instrument', 'eval_tissue']


def symlink_force(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    for name in SOURCES:
        eval_dir = CONTINUOUS_ROOT / name
        old_split = eval_dir / 'ImageSets' / 'val.txt'
        lines = [l.strip() for l in old_split.read_text().splitlines() if l.strip()]

        new_lines = []
        n_missing = 0
        for line in lines:
            jpeg_dir, frame0 = line.split()
            stem = jpeg_dir.rstrip('/').split('/')[1]
            src_frame1 = CONTINUOUS_ROOT / stem / f'{stem}_1.png'
            if not src_frame1.exists():
                n_missing += 1
                continue
            dst_frame1 = eval_dir / jpeg_dir.rstrip('/') / f'{stem}_1.png'
            symlink_force(src_frame1, dst_frame1)
            new_lines.append(f'{jpeg_dir} {frame0} {stem}_1.png')

        out_split = eval_dir / 'ImageSets' / 'val_paired.txt'
        out_split.write_text('\n'.join(new_lines) + '\n')
        print(f'{name}: {len(new_lines)}/{len(lines)} kept ({n_missing} missing frame1) -> {out_split}')


if __name__ == '__main__':
    main()

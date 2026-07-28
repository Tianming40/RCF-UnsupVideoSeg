"""
Build a 3-frame-per-line eval split (val_triplet.txt) for
CMC_grasp0_continuous_bwdif/eval_instrument and eval_tissue, needed by
RCFTripletJointMaskModel (models/rcf_triplet_joint_mask_model.py) -- its
mask branch requires im_num==3 (asserted in _decode_head_forward)
whenever decode_head2.use_flow_feat is set, so eval needs a real 3rd
frame too, not just 2 (v121's val_paired.txt) or 1 (the base single-frame
eval).

Frame 0 stays the existing annotated frame (<stem>.png); frames 1 and 2
are the gap1 neighbours (<stem>_1.png, <stem>_2.png), symlinked in from
CMC_grasp0_continuous_bwdif/<stem>/ (same source
tools/build_paired_eval_split.py already used for frame 1).

Only frame 0 is ever scored/annotated (dataset/data.py's VideoDataset,
relaxed this session to allow frame_num>1 in eval).

Usage:
  python tools/build_triplet_eval_split.py
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
            src_frame2 = CONTINUOUS_ROOT / stem / f'{stem}_2.png'
            if not (src_frame1.exists() and src_frame2.exists()):
                n_missing += 1
                continue
            stem_dir = eval_dir / jpeg_dir.rstrip('/')
            symlink_force(src_frame1, stem_dir / f'{stem}_1.png')
            symlink_force(src_frame2, stem_dir / f'{stem}_2.png')
            new_lines.append(f'{jpeg_dir} {frame0} {stem}_1.png {stem}_2.png')

        out_split = eval_dir / 'ImageSets' / 'val_triplet.txt'
        out_split.write_text('\n'.join(new_lines) + '\n')
        print(f'{name}: {len(new_lines)}/{len(lines)} kept ({n_missing} missing frame1/frame2) -> {out_split}')


if __name__ == '__main__':
    main()

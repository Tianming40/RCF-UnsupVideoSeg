"""
Build eval_instrument/eval_tissue sets whose IMAGES come from
CMC_grasp0_continuous_bwdif (the deinterlace pass used for the multigap
dataset, tools/build_paired_multigap_dataset.py) instead of the older
CMC_grasp0_deinterlaced / CMC_grasp0_5_10_merged_bwdif pipelines.

Rationale: CMC_grasp0_continuous_bwdif was produced by a DIFFERENT bwdif
run (tools/deinterlace_cmc_grasp0_multigap.py) than the other eval sets.
Even though it's nominally "bwdif" in all cases, mixing eval images from a
different deinterlace pass than the one used to build the training data
(CMC_grasp0_multigap_paired, which symlinks straight from
CMC_grasp0_continuous_bwdif) reintroduces the same class of eval/train
mismatch this project already got burned by once this session. Only the
Annotations (segmentation masks) are reused as-is from the existing eval
sets — labels don't depend on deinterlace method, only pixel content does.

Only stems that exist in CMC_grasp0_continuous_bwdif (596 cases with a
complete 8-frame run) are included; a handful of the original eval stems
don't have a full 7-post-frame run and are dropped (see printed counts).

Usage:
  python tools/build_multigap_matched_eval.py
"""
import os
from pathlib import Path

CONTINUOUS = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
OLD_EVAL_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
OUT_ROOT = CONTINUOUS  # eval_instrument / eval_tissue built as siblings under CMC_grasp0_continuous_bwdif

SOURCES = ['eval_instrument', 'eval_tissue']


def symlink_force(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    continuous_stems = set(p.name for p in CONTINUOUS.iterdir() if p.is_dir())
    print(f'CMC_grasp0_continuous_bwdif: {len(continuous_stems)} stems available')

    for name in SOURCES:
        old_dir = OLD_EVAL_ROOT / name
        old_split = old_dir / 'ImageSets' / 'val.txt'
        lines = [l.strip() for l in old_split.read_text().splitlines() if l.strip()]
        old_stems = [l.split('/')[1] for l in lines]

        out_dir = OUT_ROOT / name
        jpeg_dir = out_dir / 'JPEGImages'
        ann_dir = out_dir / 'Annotations'
        imagesets_dir = out_dir / 'ImageSets'
        for d in (jpeg_dir, ann_dir, imagesets_dir):
            d.mkdir(parents=True, exist_ok=True)

        kept_lines = []
        n_missing = 0
        for stem, line in zip(old_stems, lines):
            src_img = CONTINUOUS / stem / f'{stem}.png'
            if not src_img.exists():
                n_missing += 1
                continue

            stem_jpeg_dir = jpeg_dir / stem
            stem_jpeg_dir.mkdir(exist_ok=True)
            symlink_force(src_img, stem_jpeg_dir / f'{stem}.png')

            src_ann_dir = old_dir / 'Annotations' / stem
            stem_ann_dir = ann_dir / stem
            stem_ann_dir.mkdir(exist_ok=True)
            for ann_file in src_ann_dir.iterdir():
                symlink_force(ann_file, stem_ann_dir / ann_file.name)

            kept_lines.append(line)

        (imagesets_dir / 'val.txt').write_text('\n'.join(kept_lines) + '\n')
        print(f'{name}: {len(kept_lines)}/{len(lines)} stems kept ({n_missing} dropped, '
              f'not in CMC_grasp0_continuous_bwdif) -> {out_dir}')


if __name__ == '__main__':
    main()

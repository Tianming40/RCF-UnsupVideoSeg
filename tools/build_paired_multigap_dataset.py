"""
Restructure the per-case multigap flow data (CMC_grasp0_continuous_bwdif +
CMC_grasp0_multigap_flows, organized "one case dir holding all 28 pairs")
into the per-PAIR directory convention VideoDataset (dataset/data.py)
actually expects (one dir = one training sample = exactly 2 frames + 1 flow
file each direction, matching the g0/g5/g10/bridge convention).

Verified against dataset/data.py directly (not assumed):
  - split line: "<jpeg_dir>/ <file0> <file1>", seq_name = last path
    component of jpeg_dir (trailing slash stripped).
  - flow path for a pair = frame1's (the SECOND frame's) full path with
    "JPEGImages" -> "Flows"+suffix and ".png" -> ".npy". So the fw flow
    file must be named <pair_name>_1.npy (matching frame1's filename
    <pair_name>_1.png), sitting in Flows_NewCT/<pair_name>/ — same pattern
    BackwardFlows_NewCT/<pair_name>/<pair_name>_1.npy for bw.

Builds directories for ALL 596 complete-7-gap cases (not just the 376
currently-unannotated ones) — which cases are actually used for training is
controlled entirely by which lines go into the split .txt, no need to
regenerate data if the eval/train boundary changes later. Symlinks (not
copies) to the already-generated source files — storage isn't a concern
either way per instruction, symlinks are simply faster to create and the
source data won't change.

Usage:
  python tools/build_paired_multigap_dataset.py
"""
import itertools
import os
from pathlib import Path

SRC_JPEG = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
SRC_FLOW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows')
OUT_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_paired')

FULL7_CASES = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/full7_cases.txt')
UNANNOTATED_CASES = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/clean_full7_unannotated.txt')


def frame_file(stem, idx):
    return f'{stem}.png' if idx == 0 else f'{stem}_{idx}.png'


def symlink_force(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    all_cases = [l.strip() for l in FULL7_CASES.read_text().splitlines() if l.strip()]
    unannotated = set(l.strip() for l in UNANNOTATED_CASES.read_text().splitlines() if l.strip())
    print(f'Building paired dataset for {len(all_cases)} cases (all with complete 8-frame set)')
    print(f'  {len(unannotated)} of these are unannotated (will go into the training split)')

    jpeg_root = OUT_ROOT / 'JPEGImages'
    fw_root = OUT_ROOT / 'Flows_NewCT'
    bw_root = OUT_ROOT / 'BackwardFlows_NewCT'
    for d in (jpeg_root, fw_root, bw_root):
        d.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(range(8), 2))  # 28 pairs
    train_lines = []
    n_pairs_built = 0
    n_missing = 0

    for stem in all_cases:
        for i, j in pairs:
            gap = j - i
            pair_name = f'{stem}_f{i}t{j}_gap{gap}'

            jpeg_dir = jpeg_root / pair_name
            fw_dir = fw_root / pair_name
            bw_dir = bw_root / pair_name

            src_i = SRC_JPEG / stem / frame_file(stem, i)
            src_j = SRC_JPEG / stem / frame_file(stem, j)
            src_fw = SRC_FLOW_ROOT / 'Flows' / stem / f'{stem}_f{i}t{j}_gap{gap}.npy'
            src_bw = SRC_FLOW_ROOT / 'BackwardFlows' / stem / f'{stem}_f{i}t{j}_gap{gap}.npy'

            if not (src_i.exists() and src_j.exists() and src_fw.exists() and src_bw.exists()):
                print(f'  [warn] {pair_name}: missing source file(s), skipping')
                n_missing += 1
                continue

            jpeg_dir.mkdir(exist_ok=True)
            fw_dir.mkdir(exist_ok=True)
            bw_dir.mkdir(exist_ok=True)

            symlink_force(src_i, jpeg_dir / f'{pair_name}.png')
            symlink_force(src_j, jpeg_dir / f'{pair_name}_1.png')
            symlink_force(src_fw, fw_dir / f'{pair_name}_1.npy')
            symlink_force(src_bw, bw_dir / f'{pair_name}_1.npy')

            n_pairs_built += 1
            if stem in unannotated:
                train_lines.append(f'JPEGImages/{pair_name}/ {pair_name}.png {pair_name}_1.png')

    imagesets_dir = OUT_ROOT / 'ImageSets'
    imagesets_dir.mkdir(exist_ok=True)
    out_split = imagesets_dir / 'train_multigap_all28.txt'
    out_split.write_text('\n'.join(sorted(train_lines)) + '\n')

    print(f'\nDone. {n_pairs_built} pair directories built ({n_missing} skipped for missing source).')
    print(f'Training split: {len(train_lines)} lines -> {out_split}')


if __name__ == '__main__':
    main()

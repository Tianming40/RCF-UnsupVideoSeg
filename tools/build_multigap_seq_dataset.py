"""
Build the "one sequence directory per case, all 8 frames listed on one split
line" dataset structure that VideoDataset's native random-gap mechanism
(dataset/data.py __getitem__, options=[1..N] with per-sample probabilities)
expects — the same layout data_medical (EndoVis-style, 225-frame continuous
sequences) already uses in production, with Flows_NewCT/_NewCT2/_NewCT3
holding gap-1/2/3 flow respectively.

This builds the SHARED underlying data for both discussed variants:
  - variant 1: gap in {1,2,3} only, reuses the existing hardcoded 3-slot
    mechanism in dataset/data.py verbatim (flow_suffix/_2/_3), zero code
    changes needed.
  - variant 2: gap in {1..7}, needs dataset/data.py's hardcoded 3-way
    elif generalized to N-way (not done by this script — this script only
    prepares the data; code changes are separate).

Both variants read from the exact same JPEGImages/ + split file this
script builds; they only differ in which Flows_gapN directories actually
get wired up via flow_suffix/flow_suffix2/flow_suffix3 (variant 1) or a
generalized per-gap list (variant 2, once dataset/data.py is extended).

Layout built under OUT_ROOT:
  JPEGImages/<stem>/            -> symlinked (whole dir) to
                                    CMC_grasp0_continuous_bwdif/<stem>/
  Flows_gap{g}/<stem>/<stem>_{j}.npy         (g = 1..7, target-frame-named,
  BackwardFlows_gap{g}/<stem>/<stem>_{j}.npy  matching data_medical's
                                               Flows_NewCT2/0002.npy convention)
  ImageSets/train_multigap_seq8_unannotated.txt
      one line per unannotated case (376 lines):
      "JPEGImages/<stem>/ <stem>.png <stem>_1.png ... <stem>_7.png"

Source data (already generated, not touched here):
  CMC_grasp0_continuous_bwdif/<stem>/{<stem>.png, <stem>_1..7.png}
  CMC_grasp0_multigap_flows/{Flows,BackwardFlows}/<stem>/<stem>_f{i}t{j}_gap{g}.npy

Usage:
  python tools/build_multigap_seq_dataset.py
"""
import os
from pathlib import Path

SRC_JPEG = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
SRC_FLOW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows')
OUT_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_seq')

FULL7_CASES = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/full7_cases.txt')
UNANNOTATED_CASES = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/clean_full7_unannotated.txt')

MAX_GAP = 7


def frame_file(stem, idx):
    return f'{stem}.png' if idx == 0 else f'{stem}_{idx}.png'


def symlink_force(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    all_cases = [l.strip() for l in FULL7_CASES.read_text().splitlines() if l.strip()]
    unannotated = set(l.strip() for l in UNANNOTATED_CASES.read_text().splitlines() if l.strip())
    print(f'{len(all_cases)} cases total, {len(unannotated)} unannotated (go into training split)')

    jpeg_root = OUT_ROOT / 'JPEGImages'
    jpeg_root.mkdir(parents=True, exist_ok=True)

    flow_roots = {}
    for g in range(1, MAX_GAP + 1):
        fw_root = OUT_ROOT / f'Flows_gap{g}'
        bw_root = OUT_ROOT / f'BackwardFlows_gap{g}'
        fw_root.mkdir(exist_ok=True)
        bw_root.mkdir(exist_ok=True)
        flow_roots[g] = (fw_root, bw_root)

    n_jpeg_linked = 0
    n_flow_pairs = {g: 0 for g in range(1, MAX_GAP + 1)}
    n_missing = 0

    for stem in all_cases:
        src_case_dir = SRC_JPEG / stem
        dst_case_dir = jpeg_root / stem
        symlink_force(src_case_dir, dst_case_dir)
        n_jpeg_linked += 1

        for g in range(1, MAX_GAP + 1):
            fw_root, bw_root = flow_roots[g]
            fw_case_dir = fw_root / stem
            bw_case_dir = bw_root / stem

            for i in range(0, 8 - g):
                j = i + g
                target_name = frame_file(stem, j)  # e.g. stem_3.png
                target_npy = target_name[:-4] + '.npy'  # stem_3.npy

                src_fw = SRC_FLOW_ROOT / 'Flows' / stem / f'{stem}_f{i}t{j}_gap{g}.npy'
                src_bw = SRC_FLOW_ROOT / 'BackwardFlows' / stem / f'{stem}_f{i}t{j}_gap{g}.npy'

                if not (src_fw.exists() and src_bw.exists()):
                    n_missing += 1
                    continue

                fw_case_dir.mkdir(exist_ok=True)
                bw_case_dir.mkdir(exist_ok=True)
                symlink_force(src_fw, fw_case_dir / target_npy)
                symlink_force(src_bw, bw_case_dir / target_npy)
                n_flow_pairs[g] += 1

    imagesets_dir = OUT_ROOT / 'ImageSets'
    imagesets_dir.mkdir(exist_ok=True)
    train_lines = []
    for stem in all_cases:
        if stem not in unannotated:
            continue
        frames = ' '.join(frame_file(stem, idx) for idx in range(8))
        train_lines.append(f'JPEGImages/{stem}/ {frames}')
    out_split = imagesets_dir / 'train_multigap_seq8_unannotated.txt'
    out_split.write_text('\n'.join(sorted(train_lines)) + '\n')

    print(f'\nJPEGImages: {n_jpeg_linked} case dirs symlinked')
    for g in range(1, MAX_GAP + 1):
        print(f'  gap{g}: {n_flow_pairs[g]} flow pairs (expected {len(all_cases) * (8 - g)})')
    print(f'  missing source flow files: {n_missing}')
    print(f'Training split: {len(train_lines)} lines -> {out_split}')


if __name__ == '__main__':
    main()

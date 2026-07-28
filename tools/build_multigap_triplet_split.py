"""
Build the split file for TripletVideoDataset (dataset/triplet_data.py):
one line per (case, i) sample, meaning frames i, i+1, i+2 of that case.
6 valid values of i per 8-frame case (0..5), only for the 376 unannotated
cases (same source list as every other multigap split this session).

Usage:
  python tools/build_multigap_triplet_split.py
"""
from pathlib import Path

UNANNOTATED_CASES = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/clean_full7_unannotated.txt')
OUT_PATH = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0/train_multigap_triplet.txt')


def main():
    cases = [l.strip() for l in UNANNOTATED_CASES.read_text().splitlines() if l.strip()]
    lines = []
    for stem in cases:
        for i in range(0, 6):  # i, i+1, i+2 all within 0..7
            lines.append(f'{stem} {i}')
    OUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f'{len(cases)} cases x 6 triplets = {len(lines)} lines -> {OUT_PATH}')


if __name__ == '__main__':
    main()

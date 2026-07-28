"""
Deinterlace the new grasp-0 multi-gap sequence (pre, post=post_1, post_2 ...
post_7 — real consecutive frames at increasing temporal distance from pre,
per the advisor's note: "post_2, post_3, ... are frames 2, 3, 4, ... away
from grasp-0/pre").

Unlike the original deinterlace_cmc_bwdif.py (which only had a 2-frame
pre/post pair per sequence, forcing a 2-input concat hack), we now have up
to 8 REAL consecutive frames per case — a genuine short sequence, not an
approximation. bwdif gets proper multi-frame temporal context (both
neighbours, not just one faked pair), processed all at once per case via
ffmpeg's filter_complex concat (NOT the concat demuxer/protocol, which was
tested and found to silently drop frames on this ffmpeg build's PTS
handling for single-image inputs — filter_complex concat with N separate
-i inputs does not have this problem, verified: `-vsync 0` output flag is
required too, without it ffmpeg's default vsync/duplicate-drop logic drops
frames even with filter_complex concat).

Scope (per explicit instruction): only the 596 cases with a COMPLETE 7-gap
frame set (pre + post_1..post_7 all present — some cases are missing later
frames because "these future frames didn't exist" per the advisor's note).
The other ~5 cases are discarded, not processed here.

detect_parity + deinterlace_or_copy logic mirrors deinterlace_cmc_bwdif.py:
idet run on the REAL 8-frame sequence (not a 2-frame proxy) to decide
TFF/BFF/Progressive, then either bwdif the whole sequence or copy through
unchanged if already progressive (matches this session's earlier finding:
576p cases are ~100% TFF, 1080p cases are ~95% progressive).

Usage:
  python tools/deinterlace_cmc_grasp0_multigap.py
"""
import subprocess
import shutil
import re
from pathlib import Path

RAW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC/grasp-0')
OUT_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
CASE_LIST = RAW_ROOT / 'full7_cases.txt'

GAPS = list(range(1, 8))  # post_1(=post) .. post_7


def frame_paths(stem):
    """Ordered list of (gap_index, src_path) for pre + post_1..post_7."""
    paths = [(0, RAW_ROOT / 'pre' / f'{stem}.png')]
    paths.append((1, RAW_ROOT / 'post' / f'{stem}_1.png'))
    for n in range(2, 8):
        paths.append((n, RAW_ROOT / f'post_{n}' / f'{stem}_{n}.png'))
    return paths


def detect_parity(paths):
    """Run idet on the real N-frame sequence, return majority label."""
    inputs = []
    for _, p in paths:
        inputs += ['-i', str(p)]
    n = len(paths)
    concat_in = ''.join(f'[{i}:v]' for i in range(n))
    filt = f'{concat_in}concat=n={n}:v=1:a=0,idet'
    cmd = ['ffmpeg', '-y', *inputs, '-filter_complex', filt, '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stderr
    matches = re.findall(
        r'Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)',
        text)
    if not matches:
        return 'Undetermined'
    tff, bff, prog, undet = map(int, matches[-1])
    counts = {'TFF': tff, 'BFF': bff, 'Progressive': prog, 'Undetermined': undet}
    return max(counts, key=counts.get)


def deinterlace_sequence(paths, out_dir, stem, parity):
    inputs = []
    for _, p in paths:
        inputs += ['-i', str(p)]
    n = len(paths)
    concat_in = ''.join(f'[{i}:v]' for i in range(n))

    if parity == 'BFF':
        bwdif_parity = 1
    else:  # TFF (default assumption if Undetermined slips through, matches original script's convention)
        bwdif_parity = 0

    filt = f'{concat_in}concat=n={n}:v=1:a=0,bwdif=mode=0:parity={bwdif_parity}:deint=0'
    out_pattern = str(out_dir / f'{stem}_TMP_%d.png')
    cmd = ['ffmpeg', '-y', *inputs, '-filter_complex', filt,
           '-vsync', '0', '-frames:v', str(n), out_pattern]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [error] {stem}: ffmpeg failed\n{result.stderr[-2000:]}')
        return False

    # rename TMP sequential outputs back to gap-indexed filenames
    for i, (gap, _) in enumerate(paths, start=1):
        tmp_path = out_dir / f'{stem}_TMP_{i}.png'
        if not tmp_path.exists():
            print(f'  [error] {stem}: expected output {tmp_path} missing')
            return False
        final_name = f'{stem}.png' if gap == 0 else f'{stem}_{gap}.png'
        tmp_path.rename(out_dir / final_name)
    return True


def copy_sequence(paths, out_dir, stem):
    for gap, src in paths:
        final_name = f'{stem}.png' if gap == 0 else f'{stem}_{gap}.png'
        shutil.copy2(src, out_dir / final_name)


def main():
    cases = [l.strip() for l in CASE_LIST.read_text().splitlines() if l.strip()]
    print(f'Processing {len(cases)} cases (complete 7-gap frame sets) -> {OUT_ROOT}')
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    stats = {'TFF': 0, 'BFF': 0, 'Progressive': 0, 'Undetermined': 0, 'error': 0}

    for i, stem in enumerate(cases):
        paths = frame_paths(stem)
        missing = [p for _, p in paths if not p.exists()]
        if missing:
            print(f'  [skip] {stem}: missing {len(missing)} source frame(s)')
            stats['error'] += 1
            continue

        out_dir = OUT_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        parity = detect_parity(paths)
        stats[parity] = stats.get(parity, 0) + 1

        if parity == 'Progressive':
            copy_sequence(paths, out_dir, stem)
        elif parity in ('TFF', 'BFF'):
            ok = deinterlace_sequence(paths, out_dir, stem, parity)
            if not ok:
                stats['error'] += 1
        else:  # Undetermined - copy through unchanged, don't guess
            print(f'  [warn] {stem}: Undetermined parity, copying unchanged')
            copy_sequence(paths, out_dir, stem)

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(cases)} done  stats={stats}')

    print(f'\nDone. Final stats: {stats}')


if __name__ == '__main__':
    main()

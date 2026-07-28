"""
Assess CMC_grasp0_multigap_flows quality via forward-backward (cycle)
consistency, using the adaptive-threshold occlusion/error detector already
implemented in utils/warp_utils.py (get_occu_mask_bidirection) -- the
Sundaram/Brox/Keutzer (ECCV 2010) formula, reused by self-supervised optical
flow literature broadly (UnFlow, SelFlow, ARFlow, etc.):

    err(x)   = || flow_fw(x) + flow_bw(x + flow_fw(x)) ||^2
    thresh(x) = scale * (||flow_fw(x)||^2 + ||flow_bw_warped(x)||^2) + bias
    inconsistent(x) = err(x) > thresh(x)

Unlike the fixed 30px-mean / 80px-p99 threshold used previously for the
g0/g5/g10/bridge "clean split" (dataset/README-documented), this threshold
is ADAPTIVE -- it scales with the flow magnitude itself, so large real
displacement (expected at large gaps) does not automatically get flagged
just for being large. This directly addresses the confound identified this
session: multigap flow magnitude grows ~linearly with gap (mean ~8px at
gap1 up to ~45px at gap7), so a fixed absolute threshold conflates "large
real motion" with "RAFT failure" more and more severely as gap grows.

For every (i,j) pair at every gap 1..7, across all 596 complete-7-gap
cases, loads the independently-generated fw/bw RAFT flow, computes the
per-pixel inconsistency mask, and reports the fraction of inconsistent
pixels per pair. Aggregates per gap: mean/median/percentiles of this
per-pair inconsistent-fraction, so gap-vs-gap quality can be compared on
equal footing (no absolute-magnitude confound).

Usage:
  python tools/multigap_flow_quality_fb_consistency.py
"""
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.warp_utils import get_occu_mask_bidirection

torch.set_num_threads(1)  # many tiny ops -- per-call thread-pool spin-up overhead dominates otherwise

FLOW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # GPU0 had ~5.4GB headroom at time of writing; inference-only, tiny tensors, no grad -- low risk to the training jobs sharing the card


def main():
    fw_root = FLOW_ROOT / 'Flows'
    bw_root = FLOW_ROOT / 'BackwardFlows'
    cases = sorted(d.name for d in fw_root.iterdir() if d.is_dir())
    print(f'{len(cases)} cases, device={DEVICE}')

    all_files = [(stem, fw_file) for stem in cases for fw_file in (fw_root / stem).glob('*.npy')]
    total = len(all_files)
    print(f'{total} fw/bw pairs to process')

    per_gap_frac = defaultdict(list)  # gap -> list of inconsistent-pixel-fraction per pair
    n_done = 0
    t0 = time.time()

    for stem, fw_file in all_files:
        name = fw_file.stem  # <stem>_f{i}t{j}_gap{g}
        gap = int(name.rsplit('_gap', 1)[1])
        bw_file = bw_root / stem / fw_file.name
        if not bw_file.exists():
            continue

        fw = np.load(fw_file).astype(np.float32)   # [H, W, 2]
        bw = np.load(bw_file).astype(np.float32)   # [H, W, 2]

        fw_t = torch.from_numpy(fw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1,2,H,W]
        bw_t = torch.from_numpy(bw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        occ = get_occu_mask_bidirection(fw_t, bw_t, scale=0.01, bias=0.5)  # [1,1,H,W]
        frac = occ.mean().item()
        per_gap_frac[gap].append(frac)
        n_done += 1

        if n_done % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (total - n_done) / rate
            print(f'  {n_done}/{total} ({100*n_done/total:.1f}%) elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s', flush=True)

    print(f'\nProcessed {n_done} fw/bw pairs\n')
    print(f'{"gap":>4} {"n":>6} {"mean_incons%":>13} {"median_incons%":>15} {"p90_incons%":>12}')
    for gap in sorted(per_gap_frac.keys()):
        arr = np.array(per_gap_frac[gap]) * 100
        print(f'{gap:>4} {len(arr):>6} {arr.mean():>13.2f} {np.median(arr):>15.2f} {np.percentile(arr,90):>12.2f}')

    out_path = Path('/tmp/claude-1011/-media-mitiadmin-Micron-7450-1-tianming-RCF-UnsupVideoSeg/967cb7c3-a9f6-4c1e-b12e-d38159ada2cc/scratchpad/gap_fb_consistency.npz')
    np.savez(out_path, **{f'gap{g}': np.array(v) for g, v in per_gap_frac.items()})
    print(f'\nSaved per-pair fractions to {out_path}')


if __name__ == '__main__':
    main()

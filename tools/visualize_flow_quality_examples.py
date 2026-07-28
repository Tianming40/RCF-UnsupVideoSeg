"""
Pick concrete high-inconsistency (and typical/median) multigap flow examples
across gap=1/4/7 and render frame pair + flow color + FB-occlusion-mask
overlay, so the failure mode driving the inconsistency numbers
(tools/multigap_flow_quality_fb_consistency.py) can actually be looked at,
not just guessed at from domain reasoning.

Usage:
  python tools/visualize_flow_quality_examples.py
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import flow_vis

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.warp_utils import get_occu_mask_bidirection

FLOW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows')
JPEG_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
OUT_DIR = Path('/tmp/claude-1011/-media-mitiadmin-Micron-7450-1-tianming-RCF-UnsupVideoSeg/967cb7c3-a9f6-4c1e-b12e-d38159ada2cc/scratchpad/flow_quality_vis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def frame_file(stem, idx):
    return f'{stem}.png' if idx == 0 else f'{stem}_{idx}.png'


def load_pair_and_flow(stem, i, j, gap):
    fw = np.load(FLOW_ROOT / 'Flows' / stem / f'{stem}_f{i}t{j}_gap{gap}.npy').astype(np.float32)
    bw = np.load(FLOW_ROOT / 'BackwardFlows' / stem / f'{stem}_f{i}t{j}_gap{gap}.npy').astype(np.float32)
    img_i = Image.open(JPEG_ROOT / stem / frame_file(stem, i)).convert('RGB')
    img_j = Image.open(JPEG_ROOT / stem / frame_file(stem, j)).convert('RGB')
    return img_i, img_j, fw, bw


def render_example(stem, i, j, gap, out_name):
    img_i, img_j, fw, bw = load_pair_and_flow(stem, i, j, gap)

    fw_t = torch.from_numpy(fw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    bw_t = torch.from_numpy(bw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    occ = get_occu_mask_bidirection(fw_t, bw_t, scale=0.01, bias=0.5)[0, 0].cpu().numpy()  # [H,W] in {0,1}
    frac = occ.mean()

    flow_color = flow_vis.flow_to_color(fw, convert_to_bgr=False)  # [H,W,3] uint8

    img_i_np = np.array(img_i.resize((fw.shape[1], fw.shape[0])))
    img_j_np = np.array(img_j.resize((fw.shape[1], fw.shape[0])))

    # occlusion overlay: red where inconsistent, on top of frame i
    overlay = img_i_np.copy()
    red = np.zeros_like(overlay); red[..., 0] = 255
    alpha = (occ[..., None] * 0.55)
    overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)

    # stack: frame_i | frame_j | flow_color | occlusion_overlay
    row = np.concatenate([img_i_np, img_j_np, flow_color, overlay], axis=1)
    out_path = OUT_DIR / f'{out_name}_frac{frac*100:.1f}pct.png'
    Image.fromarray(row).save(out_path)
    print(f'{out_name}: stem={stem} i={i} j={j} gap={gap} inconsistent_frac={frac*100:.1f}% -> {out_path}')
    return frac


def main():
    fw_root = FLOW_ROOT / 'Flows'
    cases = sorted(d.name for d in fw_root.iterdir() if d.is_dir())

    for gap in (1, 4, 7):
        entries = []  # (frac, stem, i, j)
        for stem in cases:
            for i in range(0, 8 - gap):
                j = i + gap
                fw_file = fw_root / stem / f'{stem}_f{i}t{j}_gap{gap}.npy'
                bw_file = FLOW_ROOT / 'BackwardFlows' / stem / f'{stem}_f{i}t{j}_gap{gap}.npy'
                if not (fw_file.exists() and bw_file.exists()):
                    continue
                fw = np.load(fw_file).astype(np.float32)
                bw = np.load(bw_file).astype(np.float32)
                fw_t = torch.from_numpy(fw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                bw_t = torch.from_numpy(bw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                occ = get_occu_mask_bidirection(fw_t, bw_t, scale=0.01, bias=0.5)
                frac = occ.mean().item()
                entries.append((frac, stem, i, j))

        entries.sort(key=lambda x: x[0])
        n = len(entries)
        median_entry = entries[n // 2]
        high_entry = entries[-1]  # worst case
        p90_entry = entries[int(n * 0.9)]

        print(f'\n=== gap{gap}: n={n} median_frac={median_entry[0]*100:.1f}% p90_frac={p90_entry[0]*100:.1f}% max_frac={high_entry[0]*100:.1f}% ===')
        render_example(median_entry[1], median_entry[2], median_entry[3], gap, f'gap{gap}_median')
        render_example(p90_entry[1], p90_entry[2], p90_entry[3], gap, f'gap{gap}_p90')
        render_example(high_entry[1], high_entry[2], high_entry[3], gap, f'gap{gap}_worst')


if __name__ == '__main__':
    main()

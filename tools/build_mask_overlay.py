import numpy as np
from PIL import Image
import glob, os, re

INST_DIR   = '/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/saved/eval_top4_visualize_260708_151457/v83/epoch=8-step=3501/inst_out/eval_out'
TISSUE_DIR = '/media/mitiadmin/Micron_7450_1/tianming/RCF-UnsupVideoSeg/saved/eval_top4_visualize_260708_151457/v83/epoch=8-step=3501/tissue_out/eval_out'
OUT_DIR    = '/tmp/claude-1011/-media-mitiadmin-Micron-7450-1-tianming-RCF-UnsupVideoSeg/967cb7c3-a9f6-4c1e-b12e-d38159ada2cc/scratchpad/v83_overlay/out'
os.makedirs(OUT_DIR, exist_ok=True)

CYAN = np.array([0, 200, 255])    # instrument (matches grasp0_annotation_vis.py's "grasp point" color)
LIME = np.array([57, 255, 20])    # tissue     (matches grasp0_annotation_vis.py's "dissection point" color)
ALPHA = 0.45

def case_id(fname):
    # eval_{case}_{seqid}_{case}_{frame}.jpg -> case
    m = re.match(r'eval_(.+?)_\d+_\1_\d+\.jpg$', os.path.basename(fname))
    return m.group(1) if m else None

def load_rows(path):
    im = np.array(Image.open(path).convert('RGB'))
    H, W = im.shape[:2]
    row_h, col_w = H // 8, W // 2
    orig = im[0:row_h, 0:col_w]                                   # Row0 left col
    final_mask = im[row_h*7:row_h*8, col_w:2*col_w]                # Row7 right col (GF union)
    return orig, final_mask, row_h, col_w

inst_files = {case_id(f): f for f in glob.glob(f'{INST_DIR}/*.jpg') if case_id(f)}
tissue_files = {case_id(f): f for f in glob.glob(f'{TISSUE_DIR}/*.jpg') if case_id(f)}
common = sorted(set(inst_files) & set(tissue_files))
print(f'inst files: {len(inst_files)}  tissue files: {len(tissue_files)}  matched: {len(common)}')

N = min(100, len(common))
for case in common[:N]:
    orig, inst_mask_rgb, row_h, col_w = load_rows(inst_files[case])
    _, tissue_mask_rgb, _, _ = load_rows(tissue_files[case])

    inst_bin = (inst_mask_rgb.mean(axis=-1) > 127)
    tissue_bin = (tissue_mask_rgb.mean(axis=-1) > 127)

    overlay = orig.copy().astype(np.float32)
    overlay[tissue_bin] = overlay[tissue_bin] * (1 - ALPHA) + LIME * ALPHA
    overlay[inst_bin]   = overlay[inst_bin]   * (1 - ALPHA) + CYAN * ALPHA   # inst drawn last -> wins on overlap
    overlay = overlay.astype(np.uint8)

    combined = np.concatenate([orig, overlay], axis=0)   # 2-row image: original / overlay
    Image.fromarray(combined).save(f'{OUT_DIR}/{case}.png')

print(f'done, {N} overlay images saved to {OUT_DIR}')

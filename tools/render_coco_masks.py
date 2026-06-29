"""
从 COCO 格式标注渲染二值 mask，覆盖写回 eval_tissue / eval_instrument 的 Annotations 目录。
cat 1 = Soft Tissue, cat 2 = Instrument
tissue mask 会把 instrument 区域抠掉。
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

DATASET = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced')
ANNO    = DATASET / 'annotations/instances_default.json'

OUT_DIR = DATASET / 'masks_all'
OUT_DIR.mkdir(exist_ok=True)

coco = COCO(str(ANNO))
cat_tissue = 1   # Soft Tissue
cat_inst   = 2   # Instrument

for img_id, img_info in coco.imgs.items():
    stem = Path(img_info['file_name']).stem   # pre/96xxx.png -> 96xxx
    H, W = img_info['height'], img_info['width']

    # ── instrument mask ──────────────────────────────────────────────────────
    inst_mask = np.zeros((H, W), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_inst])):
        rle = coco_mask.frPyObjects(ann['segmentation'], H, W)
        inst_mask |= coco_mask.decode(coco_mask.merge(rle))

    # ── tissue mask，tissue 区域里抠掉 instrument ────────────────────────────
    soft_mask = np.zeros((H, W), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_tissue])):
        rle = coco_mask.frPyObjects(ann['segmentation'], H, W)
        soft_mask |= coco_mask.decode(coco_mask.merge(rle))
    soft_mask[inst_mask == 1] = 0

    # ── 写到 eval_instrument ─────────────────────────────────────────────────
    inst_dir = DATASET / 'eval_instrument' / 'Annotations' / stem
    if inst_dir.exists():
        Image.fromarray(inst_mask * 255).convert('L').save(inst_dir / f'{stem}.jpg')

    # ── 写到 eval_tissue ─────────────────────────────────────────────────────
    tissue_dir = DATASET / 'eval_tissue' / 'Annotations' / stem
    if tissue_dir.exists():
        Image.fromarray(soft_mask * 255).convert('L').save(tissue_dir / f'{stem}.jpg')

    # ── 统一输出到 masks_all/ ─────────────────────────────────────────────────
    Image.fromarray(inst_mask * 255).convert('L').save(OUT_DIR / f'{stem}_instrument.png')
    Image.fromarray(soft_mask * 255).convert('L').save(OUT_DIR / f'{stem}_tissue.png')

print('done')

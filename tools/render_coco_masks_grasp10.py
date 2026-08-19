"""
grasp10 version of render_coco_masks.py — merges TWO annotation batches.

Differences from the grasp0 script (verified against instances_default.json,
not assumed):
  - category IDs are SWAPPED: cat 1 = instrument, cat 2 = Soft Tissue
    (grasp0's file has cat 1 = Soft Tissue, cat 2 = instrument — do not
    copy that script's cat_tissue/cat_inst assignment as-is).
  - image file_name prefix is "pre_deinterlaced/" (grasp0 uses "pre/").
  - grasp0's eval_instrument/eval_tissue skeleton (JPEGImages/ImageSets/
    Annotations dirs) already existed before that script ran — it only
    filled in mask images, guarded by `if inst_dir.exists()`. grasp10 has
    no such skeleton yet, so this script builds it: for each annotated
    image, copies the pre-frame JPEG from JPEGImages/<stem>/<stem>.png into
    eval_instrument|eval_tissue/JPEGImages/<stem>/, creates the Annotations
    dir, and writes ImageSets/val.txt listing every annotated stem (same
    line format as grasp0's val.txt: "JPEGImages/<stem>/ <stem>.png").

Two annotation batches, verified disjoint (zero stem overlap, 103+103=206
images total):
  annotations/instances_default.json          — 103 images, has BOTH
                                                  instrument (cat 1) and
                                                  tissue (cat 2) annotations.
  annotations_108_new/instances_default.json  — 260810: UPDATED batch (was
                                                  annotations_108/, instrument-
                                                  only with tissue labelling
                                                  "pending" — that placeholder
                                                  is now resolved). Same 103
                                                  images/stems as the old
                                                  annotations_108/ (verified:
                                                  identical stem set, zero
                                                  overlap with annotations/),
                                                  now carrying real tissue
                                                  (cat 2) annotations too (245
                                                  instrument + 99 tissue
                                                  annotations, vs the old
                                                  batch's 0 tissue). No code
                                                  change needed beyond the
                                                  path below -- an all-zero
                                                  soft_mask only happens now
                                                  for images that genuinely
                                                  have zero cat-2 annotations
                                                  in the new file, not as a
                                                  blanket placeholder.

Same core logic as grasp0: tissue mask has the instrument-overlap region
cut out (instrument occludes tissue where they overlap) — trivially
satisfied for the all-black annotations_108 case since there's nothing to
cut from an already-empty mask.
"""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

DATASET = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced')
ANNO_FILES = [
    DATASET / 'annotations/instances_default.json',
    DATASET / 'annotations_108_new/instances_default.json',
]
JPEG_SRC = DATASET / 'JPEGImages'

OUT_DIR = DATASET / 'masks_all'
OUT_DIR.mkdir(exist_ok=True)

EVAL_INST = DATASET / 'eval_instrument'
EVAL_TISSUE = DATASET / 'eval_tissue'
for d in (EVAL_INST, EVAL_TISSUE):
    (d / 'JPEGImages').mkdir(parents=True, exist_ok=True)
    (d / 'Annotations').mkdir(parents=True, exist_ok=True)
    (d / 'ImageSets').mkdir(parents=True, exist_ok=True)

cat_inst   = 1   # instrument (verified from categories list, NOT grasp0's order)
cat_tissue = 2   # Soft Tissue

val_lines = []
seen_stems = set()

for anno_path in ANNO_FILES:
    print(f'--- processing {anno_path} ---')
    coco = COCO(str(anno_path))

    for img_id, img_info in coco.imgs.items():
        stem = Path(img_info['file_name']).stem   # pre_deinterlaced/96xxx.png -> 96xxx
        if stem in seen_stems:
            print(f'  [warn] {stem}: duplicate stem across annotation batches, skipping second occurrence')
            continue
        H, W = img_info['height'], img_info['width']

        src_png = JPEG_SRC / stem / f'{stem}.png'
        if not src_png.exists():
            print(f'  [warn] {stem}: source JPEG not found at {src_png}, skipping')
            continue

        # ── instrument mask ──────────────────────────────────────────────────
        inst_mask = np.zeros((H, W), dtype=np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_inst])):
            rle = coco_mask.frPyObjects(ann['segmentation'], H, W)
            inst_mask |= coco_mask.decode(coco_mask.merge(rle))

        # ── tissue mask, instrument-overlap cut out ────────────────────────
        # (all-zero for annotations_108 stems — no cat-2 annotations there)
        soft_mask = np.zeros((H, W), dtype=np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_tissue])):
            rle = coco_mask.frPyObjects(ann['segmentation'], H, W)
            soft_mask |= coco_mask.decode(coco_mask.merge(rle))
        soft_mask[inst_mask == 1] = 0

        # ── build skeleton + write masks for eval_instrument ───────────────
        inst_jpeg_dir = EVAL_INST / 'JPEGImages' / stem
        inst_jpeg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_png, inst_jpeg_dir / f'{stem}.png')
        inst_ann_dir = EVAL_INST / 'Annotations' / stem
        inst_ann_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(inst_mask * 255).convert('L').save(inst_ann_dir / f'{stem}.jpg')

        # ── build skeleton + write masks for eval_tissue ────────────────────
        tissue_jpeg_dir = EVAL_TISSUE / 'JPEGImages' / stem
        tissue_jpeg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_png, tissue_jpeg_dir / f'{stem}.png')
        tissue_ann_dir = EVAL_TISSUE / 'Annotations' / stem
        tissue_ann_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(soft_mask * 255).convert('L').save(tissue_ann_dir / f'{stem}.jpg')

        # ── unified backup ───────────────────────────────────────────────────
        Image.fromarray(inst_mask * 255).convert('L').save(OUT_DIR / f'{stem}_instrument.png')
        Image.fromarray(soft_mask * 255).convert('L').save(OUT_DIR / f'{stem}_tissue.png')

        val_lines.append(f'JPEGImages/{stem}/ {stem}.png')
        seen_stems.add(stem)

for d in (EVAL_INST, EVAL_TISSUE):
    with open(d / 'ImageSets' / 'val.txt', 'w') as f:
        f.write('\n'.join(sorted(val_lines)) + '\n')

print(f'done: {len(val_lines)} annotated images rendered to eval_instrument / eval_tissue (merged from {len(ANNO_FILES)} batches)')

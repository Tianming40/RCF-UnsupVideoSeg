"""
Quick diagnostic: how consistent is the CURRENT per-frame-independent mask
prediction (v102 checkpoint) across adjacent frames of the same real
continuous sequence (CMC_grasp0_multigap_seq / CMC_grasp0_continuous_bwdif)?

Motivation: before investing in a bigger architecture change (joint
2-frame mask prediction via cross-frame attention, replacing/augmenting the
residual bridge), check how much headroom there actually is -- if the
existing single-frame mask is already fairly stable frame-to-frame despite
being predicted independently, a cross-frame redesign has less to gain.

Uses tools/maa_union_inference.py's existing load_config/build_model/
load_checkpoint/RCFFeatureExtractor helpers (no new model-loading code).
Runs a handful of cases (single-frame forward passes only, no sliding
window -- this is a relative/internal consistency comparison, not meant to
reproduce official eval numbers) and reports mean IoU between adjacent
frames (gap1) vs far-apart frames (gap7, frame0 vs frame7) for the
instrument channel (channel 1) and tissue channel (channel 2).

Usage:
  python tools/mask_temporal_consistency_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

from tools.maa_union_inference import load_config, load_checkpoint, RCFFeatureExtractor, _get_transform
from models.rcf_soft_tissue_model import RCFSoftTissueModel


def build_model(cfg: dict, output_dir: str = "/tmp"):
    import argparse as _ap, copy
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=True, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    cfg = copy.deepcopy(cfg)
    cfg["model_kwargs"]["allow_mask_resize"] = True
    return RCFSoftTissueModel(args=fake_args, **cfg["model_kwargs"])

DEVICE = 'cpu'  # GPUs occupied by live training jobs; this is a handful of forward passes, CPU is fine

CONFIG = 'configs/instrument/rcf_cmc_grasp0_tissue_ft_v102.yaml'
CKPT = 'saved_discrete_data/grasp0_tissue_ft_v102_260709_192435/epoch=27-step=10892.ckpt'
CONTINUOUS_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')

N_CASES = 6
INSTRUMENT_CH = 1
TISSUE_CH = 2


def frame_file(stem, idx):
    return f'{stem}.png' if idx == 0 else f'{stem}_{idx}.png'


def iou(mask_a, mask_b):
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return None  # channel not present in either frame -- undefined, skip
    return inter / union


def main():
    cfg = load_config(CONFIG)
    model = build_model(cfg, output_dir='/tmp')
    model = load_checkpoint(model, CKPT).cpu()  # maa_union_inference.load_checkpoint moves to its own module-level DEVICE (cuda if available) -- force back to cpu
    extractor = RCFFeatureExtractor(model)
    transform = _get_transform(cfg)

    all_cases = sorted(d.name for d in CONTINUOUS_ROOT.iterdir() if d.is_dir()
                       and (d / f'{d.name}_7.png').exists())
    rng = np.random.RandomState(0)
    cases = rng.choice(all_cases, size=min(N_CASES, len(all_cases)), replace=False)
    print(f'{len(all_cases)} candidate cases, probing {len(cases)}: {list(cases)}')

    gap1_ious = {INSTRUMENT_CH: [], TISSUE_CH: []}
    gap7_ious = {INSTRUMENT_CH: [], TISSUE_CH: []}

    for stem in cases:
        masks = []  # discrete argmax mask per frame, [H, W]
        for idx in range(8):
            img = Image.open(CONTINUOUS_ROOT / stem / frame_file(stem, idx)).convert('RGB')
            item = {'imgs': [img], 'ann': Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))}
            item = transform(item)
            img_t = item['imgs'][0].unsqueeze(0)  # [1, 3, H, W] -- single frame, no residual head needed
            with torch.no_grad():
                feat = model.extract_feat(img_t, model.backbone2)
                mask_logits = model._decode_head_forward(feat, model.decode_head2)
                soft_mask = F.softmax(mask_logits, dim=1)[0]  # [C, fh, fw]
            discrete = soft_mask.argmax(dim=0).numpy()  # [fh, fw]
            masks.append(discrete)

        for ch in (INSTRUMENT_CH, TISSUE_CH):
            for i in range(7):
                a = masks[i] == ch
                b = masks[i + 1] == ch
                v = iou(a, b)
                if v is not None:
                    gap1_ious[ch].append(v)
            a7 = masks[0] == ch
            b7 = masks[7] == ch
            v7 = iou(a7, b7)
            if v7 is not None:
                gap7_ious[ch].append(v7)

        print(f'{stem}: done')

    print('\n=== Frame-to-frame mask consistency (IoU), v102 checkpoint, argmax channel ===')
    for ch, name in ((INSTRUMENT_CH, 'instrument'), (TISSUE_CH, 'tissue')):
        g1 = np.array(gap1_ious[ch])
        g7 = np.array(gap7_ious[ch])
        print(f'{name}: gap1 (adjacent) IoU mean={g1.mean():.3f} median={np.median(g1):.3f} n={len(g1)}'
              f' | gap7 (frame0 vs frame7) IoU mean={g7.mean():.3f} median={np.median(g7):.3f} n={len(g7)}')


if __name__ == '__main__':
    main()

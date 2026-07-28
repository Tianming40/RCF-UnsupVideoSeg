"""
TripletVideoDataset: loads 3 CONSECUTIVE frames (frame_i, frame_j=i+1,
frame_k=i+2) per sample, plus all 3 pairwise flows between them
(i->j gap1, j->k gap1, i->k gap2, each fw+bw = 6 flow fields total) --
reading directly from the already-generated CMC_grasp0_continuous_bwdif
(frames) and CMC_grasp0_multigap_flows (flow) directories, no new
symlinked directory structure needed.

New, standalone dataset class -- does NOT subclass or modify VideoDataset
(dataset/data.py), which structurally only supports a single flow pair per
sample (FlowTransform's own `assert len(data[key]) == 1` check, baked in
for the 2-frame case). Reuses the SAME underlying transform building
blocks (ResolutionAwareCrop, RandomFlip, PhotoMetricDistortion,
FlowTransform, NumpyToTensor, TorchNormalize -- all from dataset/
transforms.py, none of them modified) via build_triplet_transform() below,
just with 6 flow field names instead of 2 -- every one of those transform
classes already loops over `results.get('seg_fields', [])` generically, so
this works without touching any of them.

Split file format: one line per sample, "<stem> <i>" (whitespace-separated),
meaning the triplet is frames i, i+1, i+2 of that case (see
tools/build_multigap_triplet_split.py).
"""
import torch
import torch.utils.data
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as tv_transforms

from dataset.transforms import (
    ResolutionAwareCrop, RandomFlip, PhotoMetricDistortion,
    FlowTransform, NumpyToTensor, TorchNormalize,
)

CONTINUOUS_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif')
FLOW_ROOT = Path('/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows')

# Keys used for seg_fields -- one pair of (fw, bw) per pairwise relationship.
FLOW_KEYS_IJ = ('flow_ij_fw', 'flow_ij_bw')
FLOW_KEYS_JK = ('flow_jk_fw', 'flow_jk_bw')
FLOW_KEYS_IK = ('flow_ik_fw', 'flow_ik_bw')
ALL_FLOW_KEYS = FLOW_KEYS_IJ + FLOW_KEYS_JK + FLOW_KEYS_IK


def build_triplet_transform(strong_aug=True, resolution_crop_configs=None,
                            resize_short=400, crop_size=None):
    """Same building blocks as dataset.transforms.Transform, reused as-is,
    just wired for 6 flow field names instead of 2. flow_drop semantics for
    RandomFlip: pass ALL_FLOW_KEYS so every one of the 6 fields gets its x
    (or y) channel correctly negated on flip -- see dataset/transforms.py's
    RandomFlip docstring (this session's flip-sign fix)."""
    if resolution_crop_configs:
        resize_crop = [ResolutionAwareCrop(resolution_crop_configs)]
    else:
        from dataset.transforms import Resize, RandomCrop
        _crop = tuple(crop_size) if crop_size else (384, 384)
        resize_crop = [
            Resize(img_scale=(9999, resize_short), ratio_range=(0.96, 1.0)),
            RandomCrop(crop_size=_crop, cat_max_ratio=1.0),
        ]
    normalize_kwargs = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    return tv_transforms.Compose([
        *resize_crop,
        *([
            RandomFlip(flip_ratio=0.5, direction='horizontal', flow_fields=ALL_FLOW_KEYS),
            PhotoMetricDistortion(),
        ] if strong_aug else []),
        FlowTransform(list(ALL_FLOW_KEYS), scale_flow=False),
        NumpyToTensor(['img']),
        TorchNormalize(**normalize_kwargs),
    ])


def frame_file(stem, idx):
    return f'{stem}.png' if idx == 0 else f'{stem}_{idx}.png'


class TripletVideoDataset(torch.utils.data.Dataset):
    def __init__(self, split_path, transform, training=True):
        super().__init__()
        self.training = training
        assert training, "TripletVideoDataset is train-only (no eval/annotation path implemented)"
        lines = [l.strip() for l in open(split_path) if l.strip()]
        self.samples = []  # (stem, i)
        for line in lines:
            stem, i = line.split()
            self.samples.append((stem, int(i)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_flow_pair(self, stem, a, b):
        gap = b - a
        fw = np.load(FLOW_ROOT / 'Flows' / stem / f'{stem}_f{a}t{b}_gap{gap}.npy').astype(np.float32)
        bw = np.load(FLOW_ROOT / 'BackwardFlows' / stem / f'{stem}_f{a}t{b}_gap{gap}.npy').astype(np.float32)
        return fw, bw

    def __getitem__(self, index):
        stem, i = self.samples[index]
        j, k = i + 1, i + 2

        img_i = Image.open(CONTINUOUS_ROOT / stem / frame_file(stem, i)).convert('RGB')
        img_j = Image.open(CONTINUOUS_ROOT / stem / frame_file(stem, j)).convert('RGB')
        img_k = Image.open(CONTINUOUS_ROOT / stem / frame_file(stem, k)).convert('RGB')

        flow_ij_fw, flow_ij_bw = self._load_flow_pair(stem, i, j)
        flow_jk_fw, flow_jk_bw = self._load_flow_pair(stem, j, k)
        flow_ik_fw, flow_ik_bw = self._load_flow_pair(stem, i, k)

        ret = {
            'img': [np.asarray(img_i), np.asarray(img_j), np.asarray(img_k)],
            'seg_fields': list(ALL_FLOW_KEYS),
            'flow_ij_fw': [flow_ij_fw], 'flow_ij_bw': [flow_ij_bw],
            'flow_jk_fw': [flow_jk_fw], 'flow_jk_bw': [flow_jk_bw],
            'flow_ik_fw': [flow_ik_fw], 'flow_ik_bw': [flow_ik_bw],
            'seq_names': stem,
            'seq_ids': index,
            'paths': [str(CONTINUOUS_ROOT / stem / frame_file(stem, x)) for x in (i, j, k)],
        }
        ret = self.transform(ret)
        ret['imgs'] = ret.pop('img')
        # Drop transform-pipeline bookkeeping keys not needed downstream and
        # not guaranteed collate-safe (e.g. scale_idx can be None) -- same
        # spirit as Transform.__call__'s own `data.pop('scale_idx')`, just
        # more thorough since this dataset doesn't go through that class.
        for k in ('scale', 'scale_idx', 'img_shape', 'pad_shape',
                 'scale_factor', 'keep_ratio', 'flip', 'flip_direction'):
            ret.pop(k, None)
        return ret

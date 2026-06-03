#!/usr/bin/env python3
"""
MAA + Flow Weighted Inference Ensemble — with flow area normalization.

Differences from maa_inference_ensemble.py:
  - Flow score is area-normalized to remove the size bias that causes the scorer
    to trivially prefer smaller (higher-threshold) masks:
        flow_score_norm = flow_score_raw + area_beta * log(frac + eps)
    log(frac) is always negative; smaller masks get a larger penalty.
    area_beta=0 disables normalization (recovers original behaviour).
    Typical useful range: 1.0–3.0. Default: 1.0 (conservative start).
  - frac (mask area fraction) is logged per threshold in scores.txt so you can
    verify actual values and tune area_beta accordingly.
  - CombinedEnsemble returns best_th binary directly when --use_argmax is set
    (simpler path; nearly equivalent to softmax blend with T=10 anyway).

For each frame pair (frame + next frame):
  1. Extract backbone features ONCE
  2. Appearance path  → soft_mask [C, H, W]  (decode_head2, FCN)
  3. For each threshold candidate:
       a. Apply threshold → binary candidate mask
       b. Flow score (raw): run decode_head(mask, flow) → reconstruction error
                            flow_score_raw = -error
          Flow score (normalised): flow_score_raw + area_beta * log(frac + eps)
       c. MAA score: DINO ViT NCut on the candidate mask
       d. combined  = alpha * maa_score + (1-alpha) * flow_score_norm
  4. Softmax-weighted average of candidates (or argmax) → final soft mask
  5. Binarize and save

Usage (with flow + area normalisation):
  python tools/maa_inference_ensemble_areaNorm.py \\
    --config  configs/instrument/rcf_cmc_all_finetune_v2.yaml \\
    --ckpt    saved/.../epoch=8-step=2161.ckpt \\
    --output  saved/maa_areaNorm_output \\
    --use_flow \\
    --flow_area_beta 1.0

Disable normalisation to reproduce original behaviour:
    --flow_area_beta 0.0
"""

import argparse
import math
import sys
import yaml
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "SemanticConstraintsAndMAA"))

# ── monkey-patch V2 class (same as main_v2.py) ──────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

from models.rcf_model import RCFModel
from models.dino_vit import get_dino_model
from dataset.data import VideoDataset
from maa import soft_ncut_value

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════════════
# 1.  Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, output_dir: str = "/tmp/maa_inference") -> RCFModel:
    import argparse as _ap
    import copy
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False,
        eval_export=False,
        export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=cfg.get("object_channel", None),
        log_interval=9999,
    )
    return RCFModel(args=fake_args, **copy.deepcopy(cfg["model_kwargs"]))


def load_checkpoint(model: RCFModel, ckpt_path: str) -> RCFModel:
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()
              if k.startswith("model.")}
    mis = model.load_state_dict(sd, strict=False)
    logger.info(f"Mismatches: {mis}")
    return model.eval().to(DEVICE)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Feature extractor
# ════════════════════════════════════════════════════════════════════════════

class RCFFeatureExtractor:
    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def extract(self, batch: dict) -> dict:
        imgs_list = batch["imgs"]
        imgs = torch.stack(imgs_list, dim=1).to(DEVICE)
        B, im_num, C3, H, W = imgs.shape
        img_flat = imgs.view(B * im_num, C3, H, W)

        all_feat = self.m.extract_feat(img_flat, self.m.backbone2)

        mask_pre = self.m._decode_head_forward(all_feat, self.m.decode_head2)
        _, _, fh, fw = mask_pre.shape
        soft_mask = mask_pre.view(B, im_num, self.m.mask_layer, fh, fw)
        soft_mask = F.softmax(soft_mask, dim=2)

        if self.m.separate_residual:
            res_fw, res_bw = self.m.pred_separate_residual(all_feat, B, im_num)
        else:
            res_fw = res_bw = None

        return dict(imgs=imgs, soft_mask=soft_mask,
                    res_fw=res_fw, res_bw=res_bw,
                    B=B, im_num=im_num)


# ════════════════════════════════════════════════════════════════════════════
# 2b. TTA helper
# ════════════════════════════════════════════════════════════════════════════

def extract_with_tta(extractor: RCFFeatureExtractor, batch: dict) -> dict:
    orig = extractor.extract(batch)
    flipped_imgs = [torch.flip(img, dims=[-1]) for img in batch["imgs"]]
    flip = extractor.extract({**batch, "imgs": flipped_imgs})
    flipped_mask_back = torch.flip(flip["soft_mask"], dims=[-1])
    avg_mask = (orig["soft_mask"] + flipped_mask_back) / 2.0
    return {**orig, "soft_mask": avg_mask}


# ════════════════════════════════════════════════════════════════════════════
# 2c. Multi-checkpoint ensemble
# ════════════════════════════════════════════════════════════════════════════

def extract_multi_ckpt(extractors: list, batch: dict, use_tta: bool) -> dict:
    all_masks = []
    primary = None
    for i, ext in enumerate(extractors):
        internals = extract_with_tta(ext, batch) if use_tta else ext.extract(batch)
        all_masks.append(internals["soft_mask"])
        if i == 0:
            primary = internals
    avg_mask = torch.stack(all_masks, dim=0).mean(dim=0)
    return {**primary, "soft_mask": avg_mask}


# ════════════════════════════════════════════════════════════════════════════
# 2d. CRF post-processing
# ════════════════════════════════════════════════════════════════════════════

def build_crf_head(srgb: float = 5., sxy: float = 60.,
                   scomp: float = 5., refine_iters: int = 10,
                   crf_scale: float = 0.7):
    from models.crf_head import CRFHead
    import argparse as _ap
    fake = _ap.Namespace(checkpoints_dir="/tmp")
    head = CRFHead(args=fake, srgb=srgb, sxy=sxy,
                   scomp=scomp, refine_iters=refine_iters,
                   crf_scale=crf_scale)
    logger.info(f"CRF ready (srgb={srgb} sxy={sxy} iters={refine_iters})")
    return head


def apply_crf(crf_head, img_01: torch.Tensor,
              soft_mask: torch.Tensor) -> torch.Tensor:
    img_np = img_01[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_t  = torch.from_numpy(img_np).to(DEVICE)
    mask_t = soft_mask.to(DEVICE)
    try:
        result = crf_head.crf(img_t, mask_t)
        if not getattr(apply_crf, "_gpu_logged", False):
            logger.info("CRF running on GPU (torchcrf_cpp)")
            apply_crf._gpu_logged = True
        return result
    except Exception as e:
        logger.warning(f"GPU CRF failed ({e}), falling back to CPU")
        return crf_head.crf_cpu(img_t, mask_t)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Flow scorer
# ════════════════════════════════════════════════════════════════════════════

class FlowScorer:
    """
    Returns raw flow_score = -warp_loss (higher = better).
    Area normalization is applied in CombinedEnsemble, not here,
    so that frac is available at the call site without re-computing it.
    """

    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def score(self, internals: dict,
              gt_fw_flows: torch.Tensor,
              gt_bw_flows: torch.Tensor,
              candidate_mask: torch.Tensor) -> float:
        B      = internals["B"]
        im_num = internals["im_num"]
        imgs   = internals["imgs"]

        all_pred_mask = internals["soft_mask"].clone()
        C, fh, fw = candidate_mask.shape
        cand = F.interpolate(candidate_mask[None], size=(fh, fw),
                             mode="bilinear", align_corners=False)[0]
        all_pred_mask[:, 0] = cand.unsqueeze(0).expand(B, -1, -1, -1)

        mask_size = tuple(self.m.mask_size)
        flow_num  = gt_fw_flows.shape[1]

        def to_bflow2hw(f):
            if f.shape[-1] == 2:
                f = f.permute(0, 1, 4, 2, 3).contiguous()
            return f.to(DEVICE)

        def resize_flow(f):
            f = to_bflow2hw(f)
            H_, W_ = f.shape[-2], f.shape[-1]
            return self.m.resize(
                f.view(B * flow_num, 2, H_, W_), mask_size
            ).view(B, flow_num, 2, *mask_size)

        fw_r = resize_flow(gt_fw_flows)
        bw_r = resize_flow(gt_bw_flows)

        if self.m.allow_mask_resize and tuple(all_pred_mask.shape[-2:]) != mask_size:
            B_i, im_n, C_i, H_i, W_i = all_pred_mask.shape
            all_pred_mask = self.m.resize(
                all_pred_mask.view(B_i * im_n, C_i, H_i, W_i), mask_size
            ).view(B_i, im_n, C_i, *mask_size)

        _, loss_flow = self.m.decode_head(
            imgs, all_pred_mask, fw_r, bw_r,
            internals["res_fw"], internals["res_bw"]
        )

        seg_loss = loss_flow.get("seg",
                   loss_flow.get("seg_fw", torch.tensor(0.)) +
                   loss_flow.get("seg_bw", torch.tensor(0.)))

        val = seg_loss.item() if isinstance(seg_loss, torch.Tensor) else float(seg_loss)
        return -val


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAA scorer
# ════════════════════════════════════════════════════════════════════════════

class MAAScorer:
    def __init__(self, arch="vit_small", patch_size=8,
                 which_features="k", tau=0.2, eps=1e-5):
        self.patch_size = patch_size
        self.tau = tau
        self.eps = eps
        self.which_features = which_features

        logger.info(f"Loading DINO {arch} p={patch_size}")
        self.dino = get_dino_model(arch, patch_size, device=DEVICE)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        self._feat_out = {}
        def hook(m, i, o):
            self._feat_out["qkv"] = o
        self.dino._modules["blocks"][-1]._modules["attn"] \
            ._modules["qkv"].register_forward_hook(hook)

        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[None, :, None, None]
        self._mean, self._std = mean, std

    @torch.no_grad()
    def _get_feats(self, img_01: torch.Tensor):
        x = (img_01 - self._mean) / self._std
        h, w = x.shape[-2:]
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        x  = F.pad(x, (0, pw, 0, ph))
        h_feat = x.shape[-2] // self.patch_size
        w_feat = x.shape[-1] // self.patch_size

        self._feat_out = {}
        att = self.dino.get_last_selfattention(x)
        nb, nh, nt = att.shape[:3]
        qkv = (self._feat_out["qkv"]
               .reshape(nb, nt, 3, nh, -1 // nh)
               .permute(2, 0, 3, 1, 4))
        idx = {"k": 1, "q": 0, "v": 2}[self.which_features]
        feat = qkv[idx].transpose(1, 2).reshape(nb, nt, -1)
        return feat, h_feat, w_feat

    @torch.no_grad()
    def score(self, img_01: torch.Tensor, binary_mask: torch.Tensor) -> float:
        feats, h_feat, w_feat = self._get_feats(img_01)
        mask_feat = F.interpolate(binary_mask[None, None].float(),
                                  (h_feat, w_feat), mode="nearest")[0, 0]
        ncut = soft_ncut_value(feats, mask_feat, self.tau, self.eps)
        return float(-ncut)


# ════════════════════════════════════════════════════════════════════════════
# 5.  Combined ensemble
#     Changes vs original:
#       1. flow_norm = flow_raw + area_beta * log(frac)  — remove size bias
#       2. per-frame z-score normalisation before combining — fix scale mismatch
#          (MAA span ~0.33 vs flow_norm span ~80; raw alpha is meaningless)
#       3. argmax only — no softmax blend, no redundant eval_pos_th on soft mask
# ════════════════════════════════════════════════════════════════════════════

class CombinedEnsemble:
    """
    For each frame, sweeps thresholds, scores with MAA + area-normalised flow,
    z-score normalises both signals so alpha truly controls their balance,
    then returns the single best-scoring threshold's binary mask (argmax).

    area_beta: flow_norm = flow_raw + beta * log(frac).  0 = disabled.
    alpha:     weight of MAA in the z-normalised combined score (0–1).
               1.0 = MAA only, 0.0 = flow only.
    """

    def __init__(self, maa_scorer: MAAScorer,
                 flow_scorer: FlowScorer = None,
                 thresholds: list = None,
                 alpha: float = 0.7,
                 area_beta: float = 1.0,
                 object_channel: int = None):
        self.maa       = maa_scorer
        self.flow      = flow_scorer
        self.ths       = thresholds or [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.alpha     = alpha if flow_scorer is not None else 1.0
        self.area_beta = area_beta
        self.obj_ch    = object_channel

    @torch.no_grad()
    def combine(self, soft_mask: torch.Tensor,
                img_01: torch.Tensor,
                obj_ch: int,
                internals: dict = None,
                gt_fw_flows: torch.Tensor = None,
                gt_bw_flows: torch.Tensor = None):
        """
        Returns (best_binary [H,W], best_th float, score_log list).
        best_binary is already 0/1 — no further thresholding needed.
        """
        use_flow = (self.flow is not None and
                    internals is not None and
                    gt_fw_flows is not None)

        prob_map = soft_mask[obj_ch]
        raw = []

        for th in self.ths:
            binary = (prob_map > th).float()
            frac   = binary.mean().item()

            if frac < 0.005 or frac > 0.995:
                raw.append({"th": th, "frac": frac,
                            "maa": None, "flow_raw": None,
                            "flow_norm": None, "binary": binary})
                continue

            h_img, w_img = img_01.shape[-2:]
            binary_img = F.interpolate(binary[None, None],
                                       (h_img, w_img), mode="nearest")[0, 0]
            maa_s = self.maa.score(img_01, binary_img)

            if use_flow:
                cand = soft_mask.clone()
                cand[obj_ch] = binary
                flow_raw = self.flow.score(internals, gt_fw_flows, gt_bw_flows, cand)
                flow_norm = (flow_raw + self.area_beta * math.log(frac + 1e-8)
                             if self.area_beta != 0.0 else flow_raw)
            else:
                flow_raw = flow_norm = None

            raw.append({"th": th, "frac": frac,
                        "maa": maa_s, "flow_raw": flow_raw,
                        "flow_norm": flow_norm, "binary": binary})

        # ── Per-frame z-score normalisation ───────────────────────────────────
        # MAA span ~0.33, flow_norm span ~80 on this dataset.
        # Without normalisation, alpha is meaningless — flow always dominates.
        # Dividing each signal by its own std across thresholds equalises them
        # so alpha=0.7 truly means "70% of the decision comes from MAA".
        valid = [r for r in raw if r["maa"] is not None]

        if len(valid) < 2:
            # Fallback: all thresholds were trivial; return middle threshold
            mid = raw[len(raw) // 2]
            return mid["binary"], mid["th"], []

        maa_vals  = np.array([r["maa"]       for r in valid], dtype=np.float64)
        flow_vals = np.array([r["flow_norm"]  for r in valid], dtype=np.float64) \
                    if use_flow else None

        maa_z = (maa_vals - maa_vals.mean()) / (maa_vals.std() + 1e-8)

        if use_flow and flow_vals is not None:
            flow_z = (flow_vals - flow_vals.mean()) / (flow_vals.std() + 1e-8)
            alpha  = self.alpha
            combined = alpha * maa_z + (1.0 - alpha) * flow_z
        else:
            combined = maa_z
            alpha    = 1.0

        # ── Argmax: pick the single best threshold ────────────────────────────
        best_idx = int(np.argmax(combined))
        best     = valid[best_idx]

        log = [{
            "th":        r["th"],
            "frac":      round(r["frac"], 4),
            "maa":       round(r["maa"], 4),
            "maa_z":     round(float(maa_z[i]), 4),
            "flow_raw":  round(r["flow_raw"],  4) if r["flow_raw"]  is not None else None,
            "flow_norm": round(r["flow_norm"], 4) if r["flow_norm"] is not None else None,
            "flow_z":    round(float(flow_z[i]), 4) if use_flow else None,
            "combined":  round(float(combined[i]), 4),
        } for i, r in enumerate(valid)]
        log.append({"alpha": round(alpha, 4)})

        return best["binary"], best["th"], log


# ════════════════════════════════════════════════════════════════════════════
# 6.  Dataset + utilities  (unchanged from original)
# ════════════════════════════════════════════════════════════════════════════

class _TransformWithDummyAnn:
    def __init__(self, transform):
        self._t = transform

    def __call__(self, data):
        if 'ann' not in data:
            data['ann'] = Image.fromarray(
                np.zeros((1, 1, 3), dtype=np.uint8))
        return self._t(data)


def build_dataset(cfg: dict, use_flow: bool) -> VideoDataset:
    import dataset as _dataset_mod
    import argparse as _ap

    fake_args = _ap.Namespace(
        train_transform_kwargs=cfg.get("train_transform_kwargs", {"strong_aug": True}),
        test_transform_kwargs=cfg.get("test_transform_kwargs", {"strong_aug": False}),
    )
    base_transform = _dataset_mod.get_transform(fake_args, training=False)

    test_kw   = cfg.get("test_dataset_kwargs", {})
    data_path = cfg.get("test_data_path") or cfg.get("data_path")
    flow_suffix = test_kw.get(
        "flow_suffix",
        cfg.get("train_dataset_kwargs", {}).get("flow_suffix", "_NewCT")
    )

    if use_flow:
        transform   = _TransformWithDummyAnn(base_transform)
        ds_training = True
    else:
        transform   = base_transform
        ds_training = False

    return VideoDataset(
        root=data_path,
        split=test_kw.get("split", "val.txt"),
        training=ds_training,
        frame_num=2 if use_flow else 1,
        load_flow=use_flow,
        flow_suffix=flow_suffix,
        zero_ann=test_kw.get("zero_ann", True),
        transform=transform,
    )


def to_device(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(DEVICE)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(DEVICE) for t in v]
        else:
            out[k] = v
    return out


def get_img_01(batch: dict) -> torch.Tensor:
    imgs = batch["imgs"]
    img  = (imgs[0] if isinstance(imgs, list) else imgs[:, 0]).float()
    return ((img + 2.0) / 4.0).clamp(0, 1)[:1]


def save_pair(img_01: torch.Tensor, soft_mask: torch.Tensor,
              out_dir: Path, seq: str, fname: str, eval_pos_th: float = 0.35):
    folder = out_dir / seq
    folder.mkdir(parents=True, exist_ok=True)

    img_np = img_01[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_h, img_w = img_np.shape[:2]

    soft_full = F.interpolate(
        soft_mask[None, None].float(),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False
    )[0, 0]
    mask_full = (soft_full > eval_pos_th).cpu().numpy()

    Image.fromarray(
        (mask_full * 255).astype(np.uint8)
    ).save(str(folder / f"{fname}_mask.png"))

    overlay = img_np.astype(np.float32)
    m = mask_full
    overlay[m, 0] = overlay[m, 0] * 0.3
    overlay[m, 1] = np.clip(overlay[m, 1] * 0.5 + 180, 0, 255)
    overlay[m, 2] = overlay[m, 2] * 0.3
    Image.fromarray(overlay.astype(np.uint8)).save(
        str(folder / f"{fname}_overlay.jpg"))


# ════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ════════════════════════════════════════════════════════════════════════════

def detect_object_channel(extractor: RCFFeatureExtractor,
                           maa_scorer: MAAScorer,
                           dataset: VideoDataset,
                           n_probe: int = 10,
                           probe_th: float = 0.35) -> int:
    n_ch = extractor.m.mask_layer
    ch_scores = [[] for _ in range(n_ch)]

    total   = len(dataset)
    indices = list(np.linspace(0, total - 1, n_probe, dtype=int))
    probe_subset = torch.utils.data.Subset(dataset, indices)
    probe_loader = DataLoader(probe_subset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    logger.info(f"Probing {n_probe} frames for instrument channel ...")

    for batch in probe_loader:
        batch = to_device(batch)
        try:
            internals = extractor.extract(batch)
            soft_mask = internals["soft_mask"][0, 0]
            img_01    = get_img_01(batch)

            for ch in range(n_ch):
                prob   = soft_mask[ch]
                binary = (prob > probe_th).float()
                frac   = binary.mean().item()
                if frac < 0.005 or frac > 0.995:
                    continue
                h, w = img_01.shape[-2:]
                binary_img = F.interpolate(binary[None, None],
                                           (h, w), mode="nearest")[0, 0]
                maa_s = maa_scorer.score(img_01, binary_img)
                ch_scores[ch].append(maa_s)
        except Exception as e:
            logger.warning(f"  Probe frame skipped: {e}")
            continue

    avg_scores = [float(np.mean(v)) if v else -1e9 for v in ch_scores]
    best_ch = int(np.argmax(avg_scores))
    for ch, s in enumerate(avg_scores):
        marker = " ← selected" if ch == best_ch else ""
        logger.info(f"  channel {ch}: avg MAA = {s:.4f}{marker}")
    return best_ch


def run(args):
    cfg     = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Models ────────────────────────────────────────────────────────────────
    model = build_model(cfg, output_dir=str(out_dir))
    model = load_checkpoint(model, args.ckpt)
    extractors = [RCFFeatureExtractor(model)]

    if args.extra_ckpts:
        for ckpt_path in args.extra_ckpts:
            extra_model = build_model(cfg, output_dir=str(out_dir))
            extra_model = load_checkpoint(extra_model, ckpt_path)
            extractors.append(RCFFeatureExtractor(extra_model))
        logger.info(f"Multi-checkpoint ensemble: {len(extractors)} checkpoints")

    if args.tta:
        logger.info("TTA ENABLED (horizontal flip)")

    # ── Scorers ───────────────────────────────────────────────────────────────
    maa_scorer  = MAAScorer(arch=args.dino_arch,
                            patch_size=args.dino_patch_size,
                            tau=args.dino_tau)
    flow_scorer = FlowScorer(model) if args.use_flow else None

    crf_head = None
    if args.use_crf:
        crf_head = build_crf_head(srgb=args.crf_srgb, sxy=args.crf_sxy,
                                  refine_iters=args.crf_iters)

    if not args.use_flow:
        logger.info("Flow scoring DISABLED — using MAA only")
    else:
        logger.info(
            f"Flow scoring ENABLED  | alpha={args.alpha}  "
            f"area_beta={args.flow_area_beta}  "
            f"({'area-normalised' if args.flow_area_beta != 0 else 'original, no area norm'})"
        )
    logger.info("Output mode: argmax (best threshold binary, no softmax blend)")

    dataset = build_dataset(cfg, use_flow=args.use_flow)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    logger.info(f"Dataset: {len(dataset)} frames  "
                f"({'with' if args.use_flow else 'without'} flow)")

    # ── Detect instrument channel ─────────────────────────────────────────────
    if args.object_channel is not None:
        obj_ch = args.object_channel
        logger.info(f"Using fixed object_channel={obj_ch} from CLI")
    else:
        obj_ch = detect_object_channel(
            extractors[0], maa_scorer, dataset,
            n_probe=args.channel_probe_frames,
            probe_th=0.35)
    logger.info(f"Instrument channel: {obj_ch}")

    ensemble = CombinedEnsemble(
        maa_scorer=maa_scorer,
        flow_scorer=flow_scorer,
        thresholds=args.thresholds,
        alpha=args.alpha,
        area_beta=args.flow_area_beta,
        object_channel=obj_ch,
    )

    if not args.use_flow:
        logger.info("Flow scoring DISABLED — using MAA only (alpha=1.0)")
    else:
        logger.info(
            f"Flow scoring ENABLED | alpha(MAA)={args.alpha} "
            f"area_beta={args.flow_area_beta} | "
            "scores z-normalised per frame before combining"
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    log_lines = [
        f"instrument_channel: {obj_ch}",
        f"alpha: {args.alpha}",
        f"area_beta: {args.flow_area_beta}",
    ]

    for batch in tqdm(loader, desc="MAA+Flow(areaNorm) ensemble"):
        batch = to_device(batch)
        seq   = batch["seq_names"][0]
        fidx  = int(batch.get("frame_ind_start", [0])[0])

        internals = extract_multi_ckpt(extractors, batch, use_tta=args.tta)
        soft_mask = internals["soft_mask"][0, 0]

        gt_fw = (torch.stack(batch["gt_fw_flows"], dim=1)
                 if args.use_flow and "gt_fw_flows" in batch else None)
        gt_bw = (torch.stack(batch["gt_bw_flows"], dim=1)
                 if args.use_flow and "gt_bw_flows" in batch else None)

        img_01 = get_img_01(batch)

        final_mask, best_th, score_log = ensemble.combine(
            soft_mask, img_01, obj_ch, internals, gt_fw, gt_bw)

        fname = f"{fidx:05d}"

        # final_mask is already binary (0/1) from argmax — just upsample to img res
        if crf_head is not None:
            ih, iw = img_01.shape[-2:]
            soft_full = F.interpolate(
                final_mask[None, None].float(), (ih, iw),
                mode="bilinear", align_corners=False)[0, 0]
            soft_full = apply_crf(crf_head, img_01, soft_full)
            save_pair(img_01, soft_full, out_dir / "results", seq, fname,
                      eval_pos_th=0.5)
        else:
            save_pair(img_01, final_mask, out_dir / "results", seq, fname,
                      eval_pos_th=0.5)

        th_entries  = [s for s in score_log if "th" in s]
        alpha_entry = next((s for s in score_log if "alpha" in s), {})
        detail = " | ".join(
            f"th={s['th']:.2f} frac={s['frac']:.3f} "
            f"maa_z={s['maa_z']:.3f} flow_z={s['flow_z']:.3f} "
            f"comb={s['combined']:.3f}"
            for s in th_entries
        )
        log_lines.append(
            f"{seq}/{fname}  best_th={best_th:.2f} "
            f"alpha={alpha_entry.get('alpha', '?'):.4f}  [{detail}]")

    (out_dir / "scores.txt").write_text("\n".join(log_lines))
    logger.info(f"Done. Results: {out_dir / 'results'} | Log: {out_dir / 'scores.txt'}")


# ════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--output",  required=True)

    p.add_argument("--use_flow", action="store_true")
    p.add_argument("--flow_suffix", default="_NewCT")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="MAA weight after z-score normalisation (0=flow only, 1=MAA only)")
    p.add_argument("--flow_area_beta", type=float, default=1.0,
                   help="Area penalty: flow_norm = flow_raw + beta*log(frac). 0=disabled.")

    # TTA
    p.add_argument("--tta", action="store_true")

    # Multi-checkpoint
    p.add_argument("--extra_ckpts", nargs="*", default=[])

    # CRF
    p.add_argument("--use_crf",   action="store_true")
    p.add_argument("--crf_srgb",  type=float, default=5.)
    p.add_argument("--crf_sxy",   type=float, default=60.)
    p.add_argument("--crf_iters", type=int,   default=10)

    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    p.add_argument("--object_channel",       type=int,   default=None)
    p.add_argument("--channel_probe_frames", type=int,   default=10)

    p.add_argument("--dino_arch",       default="vit_small")
    p.add_argument("--dino_patch_size", type=int, default=8)
    p.add_argument("--dino_tau",        type=float, default=0.2)
    p.add_argument("--workers",         type=int, default=4)

    run(p.parse_args())

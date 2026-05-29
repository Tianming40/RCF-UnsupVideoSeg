#!/usr/bin/env python3
"""
MAA + Flow Weighted Inference Ensemble for RCF segmentation.

For each frame pair (frame + next frame):
  1. Extract backbone features ONCE
  2. Appearance path  → soft_mask [C, H, W]  (decode_head2, FCN)
  3. For each threshold candidate:
       a. Apply threshold → binary candidate mask
       b. Flow score  : run decode_head(mask, flow) → flow reconstruction error
                        flow_score = -error  (lower error = better mask)
       c. MAA score   : DINO ViT NCut on the candidate mask
       d. combined    = alpha * maa_score + (1-alpha) * flow_score
  4. Softmax-weighted average of candidates → final soft mask
  5. Binarize and save

If no optical flow is available (load_flow=False), falls back to MAA-only.

New files created, no existing files modified.
Reuses: models/rcf_model.py, models/dino_vit.py,
        tools/SemanticConstraintsAndMAA/maa.py

Usage (with flow):
  python tools/maa_inference_ensemble.py \\
    --config  configs/instrument/rcf_cmc_all_finetune_v2.yaml \\
    --ckpt    saved/.../epoch=8-step=2161.ckpt \\
    --output  saved/maa_ensemble_output \\
    --use_flow \\
    --flow_suffix _NewCT \\
    --alpha 0.5

Usage (MAA only, no flow):
  python tools/maa_inference_ensemble.py \\
    --config  configs/instrument/rcf_cmc_all_finetune_v2.yaml \\
    --ckpt    saved/.../epoch=8-step=2161.ckpt \\
    --output  saved/maa_ensemble_output
"""

import argparse
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
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False,
        eval_export=False,
        export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=cfg.get("object_channel", None),
        log_interval=9999,
    )
    return RCFModel(args=fake_args, **cfg["model_kwargs"])


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
# 2.  Feature extractor — runs backbone ONCE, reused for all candidates
# ════════════════════════════════════════════════════════════════════════════

class RCFFeatureExtractor:
    """
    Extracts:
      - all_feat          : backbone feature list
      - soft_mask         : [B, im_num, C, H, W]  softmax appearance mask
      - residual_fw/bw    : residual flow predictions  (None if separate_residual=False)
    """

    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def extract(self, batch: dict) -> dict:
        imgs_list = batch["imgs"]
        imgs = torch.stack(imgs_list, dim=1).to(DEVICE)  # [B, im_num, 3, H, W]
        B, im_num, C3, H, W = imgs.shape
        img_flat = imgs.view(B * im_num, C3, H, W)

        all_feat = self.m.extract_feat(img_flat, self.m.backbone2)

        # Appearance mask
        mask_pre = self.m._decode_head_forward(all_feat, self.m.decode_head2)
        _, _, fh, fw = mask_pre.shape
        soft_mask = mask_pre.view(B, im_num, self.m.mask_layer, fh, fw)
        soft_mask = F.softmax(soft_mask, dim=2)            # [B, im_num, C, fh, fw]

        # Residuals for flow path
        if self.m.separate_residual:
            res_fw, res_bw = self.m.pred_separate_residual(all_feat, B, im_num)
        else:
            res_fw = res_bw = None

        return dict(imgs=imgs, soft_mask=soft_mask,
                    res_fw=res_fw, res_bw=res_bw,
                    B=B, im_num=im_num)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Flow scorer — uses decode_head to measure how well a mask explains flow
# ════════════════════════════════════════════════════════════════════════════

class FlowScorer:
    """
    Runs decode_head(candidate_mask, flow) and returns flow_score = -loss.
    Higher score means the mask better explains the optical flow.
    """

    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def score(self, internals: dict,
              gt_fw_flows: torch.Tensor,
              gt_bw_flows: torch.Tensor,
              candidate_mask: torch.Tensor) -> float:
        """
        internals      : output of RCFFeatureExtractor.extract()
        gt_fw_flows    : [B, flow_num, 2, H, W]  from dataset
        gt_bw_flows    : [B, flow_num, 2, H, W]
        candidate_mask : [C, H, W]  binary/soft mask for one threshold

        Returns flow_score (float, higher = better mask).
        """
        B      = internals["B"]
        im_num = internals["im_num"]
        imgs   = internals["imgs"]          # [B, im_num, 3, H, W]

        # Build all_pred_mask: replace frame-0 with candidate, keep others
        all_pred_mask = internals["soft_mask"].clone()      # [B, im_num, C, fh, fw]
        C, fh, fw = candidate_mask.shape
        cand = F.interpolate(candidate_mask[None], size=(fh, fw),
                             mode="bilinear", align_corners=False)[0]  # [C, fh, fw]
        all_pred_mask[:, 0] = cand.unsqueeze(0).expand(B, -1, -1, -1)

        # Resize flows to mask_size
        mask_size = tuple(self.m.mask_size)
        flow_num  = gt_fw_flows.shape[1]

        def to_bflow2hw(f):
            """Ensure flow is [B, flow_num, 2, H, W] regardless of input layout."""
            if f.shape[-1] == 2:
                # channels last: [B, flow_num, H, W, 2] → [B, flow_num, 2, H, W]
                f = f.permute(0, 1, 4, 2, 3).contiguous()
            return f.to(DEVICE)

        def resize_flow(f):
            f = to_bflow2hw(f)   # [B, flow_num, 2, H, W]
            H_, W_ = f.shape[-2], f.shape[-1]
            return self.m.resize(
                f.view(B * flow_num, 2, H_, W_),
                mask_size
            ).view(B, flow_num, 2, *mask_size)

        fw_r = resize_flow(gt_fw_flows)
        bw_r = resize_flow(gt_bw_flows)

        # allow_mask_resize: resize needs 4D [N, C, H, W], but all_pred_mask is
        # 5D [B, im_num, C, H, W] → flatten, resize, unflatten
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
        return -val   # negate: lower loss = higher score


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAA scorer — DINO ViT NCut
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
        """
        img_01: [1, 3, H, W]  in 0-1.
        Returns (feats [1, N_tok, D], h_feat, w_feat) where h_feat*w_feat = N_tok-1.
        """
        x = (img_01 - self._mean) / self._std
        h, w = x.shape[-2:]
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        x  = F.pad(x, (0, pw, 0, ph))
        h_pad, w_pad = x.shape[-2:]
        h_feat = h_pad // self.patch_size
        w_feat = w_pad // self.patch_size

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
        """
        img_01      : [1, 3, H, W]  0-1
        binary_mask : [H, W]  0/1
        Returns MAA (higher = better).
        """
        feats, h_feat, w_feat = self._get_feats(img_01)
        # Resize mask to match padded feature map resolution
        mask_feat = F.interpolate(binary_mask[None, None].float(),
                                  (h_feat, w_feat), mode="nearest")[0, 0]
        ncut = soft_ncut_value(feats, mask_feat, self.tau, self.eps)
        return float(-ncut)


# ════════════════════════════════════════════════════════════════════════════
# 5.  Combined ensemble
# ════════════════════════════════════════════════════════════════════════════

class CombinedEnsemble:
    """
    Sweeps thresholds, scores each candidate with MAA + flow (optional),
    and returns an MAA+flow-weighted soft mask.
    """

    def __init__(self, maa_scorer: MAAScorer,
                 flow_scorer: FlowScorer = None,
                 thresholds: list = None,
                 temperature: float = 10.0,
                 alpha: float = 0.5,
                 object_channel: int = None):
        self.maa   = maa_scorer
        self.flow  = flow_scorer          # None → MAA only
        self.ths   = thresholds or [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.T     = temperature
        self.alpha = alpha if flow_scorer is not None else 1.0
        self.obj_ch = object_channel

    @torch.no_grad()
    def combine(self, soft_mask: torch.Tensor,
                img_01: torch.Tensor,
                obj_ch: int,
                internals: dict = None,
                gt_fw_flows: torch.Tensor = None,
                gt_bw_flows: torch.Tensor = None):
        """
        soft_mask : [C, H, W]  softmax probabilities
        img_01    : [1, 3, H, W]  0-1
        obj_ch    : instrument channel (determined once globally before inference)
        internals / gt_fw/bw : for flow scoring

        Sweeps thresholds only on obj_ch.
        Returns (combined_mask [H,W], best_th float, score_log list)
        """
        use_flow = (self.flow is not None and
                    internals is not None and
                    gt_fw_flows is not None)

        prob_map = soft_mask[obj_ch]   # [H, W]
        scores = []

        for th in self.ths:
            binary = (prob_map > th).float()
            frac   = binary.mean().item()

            if frac < 0.005 or frac > 0.995:
                scores.append({"th": th, "maa": -1e9,
                               "flow": -1e9, "combined": -1e9, "binary": binary})
                continue

            # MAA score
            h_img, w_img = img_01.shape[-2:]
            binary_img = F.interpolate(binary[None, None],
                                       (h_img, w_img), mode="nearest")[0, 0]
            maa_s = self.maa.score(img_01, binary_img)

            # Flow score
            if use_flow:
                cand = soft_mask.clone()
                cand[obj_ch] = binary
                flow_s = self.flow.score(internals, gt_fw_flows, gt_bw_flows, cand)
            else:
                flow_s = 0.0

            combined = self.alpha * maa_s + (1.0 - self.alpha) * flow_s
            scores.append({"th": th, "maa": maa_s, "flow": flow_s,
                           "combined": combined, "binary": binary})

        combined_vals = torch.tensor([s["combined"] for s in scores],
                                     dtype=torch.float32)
        weights = F.softmax(combined_vals * self.T, dim=0)

        final = torch.zeros_like(prob_map)
        for i, entry in enumerate(scores):
            final += weights[i].item() * entry["binary"]

        best_idx = int(combined_vals.argmax().item())
        best_th  = scores[best_idx]["th"]
        log = [{k: v for k, v in s.items() if k != "binary"} for s in scores]
        return final, best_th, log


# ════════════════════════════════════════════════════════════════════════════
# 6.  Dataset + utilities
# ════════════════════════════════════════════════════════════════════════════

class _TransformWithDummyAnn:
    """
    Wraps an existing transform and injects a dummy 1×1 zero 'ann' key
    before calling the transform. Needed when dataset runs in training=True
    mode (which does not add 'ann') but the transform pipeline expects it.
    """
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

    # When using flow, frame_num=2 is required. VideoDataset asserts frame_num==1
    # in eval (training=False) mode, so we load with training=True to bypass.
    # The model itself remains in eval() mode.
    # In training=True mode, 'ann' is not added to the batch, so we wrap the
    # transform to inject a dummy 'ann' before processing.
    if use_flow:
        transform = _TransformWithDummyAnn(base_transform)
        ds_training = True
    else:
        transform = base_transform
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
    """First frame as [1,3,H,W] in 0-1 (undo model normalisation)."""
    imgs = batch["imgs"]
    img  = (imgs[0] if isinstance(imgs, list) else imgs[:, 0]).float()
    return ((img + 2.0) / 4.0).clamp(0, 1)[:1]


def save_pair(img_01: torch.Tensor, mask_np: np.ndarray,
              out_dir: Path, seq: str, fname: str):
    """
    Save mask.png and overlay.jpg into the same folder at original image resolution.
      out_dir/<seq>/<fname>_mask.png
      out_dir/<seq>/<fname>_overlay.jpg
    Mask is upsampled to image resolution. Overlay uses bright green.
    """
    folder = out_dir / seq
    folder.mkdir(parents=True, exist_ok=True)

    # Original image at full resolution
    img_np = img_01[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_h, img_w = img_np.shape[:2]

    # Upsample mask to original image resolution
    mask_tensor = torch.from_numpy(mask_np)[None, None].float()
    mask_full = F.interpolate(mask_tensor, (img_h, img_w),
                              mode="nearest")[0, 0].numpy()

    # ── mask (binary 0/255 at full resolution) ───────────────────────────────
    Image.fromarray(
        (mask_full * 255).clip(0, 255).astype(np.uint8)
    ).save(str(folder / f"{fname}_mask.png"))

    # ── overlay (green highlight at full resolution) ─────────────────────────
    overlay = img_np.astype(np.float32)
    m = mask_full > 0.5
    overlay[m, 0] = overlay[m, 0] * 0.3                           # R dim
    overlay[m, 1] = np.clip(overlay[m, 1] * 0.5 + 180, 0, 255)   # G bright
    overlay[m, 2] = overlay[m, 2] * 0.3                           # B dim

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
    """
    Use a small DataLoader to sample n_probe frames evenly from the dataset.
    For each channel, compute average MAA at probe_th.
    Return the channel with the highest average MAA (= instrument channel).
    """
    n_ch = extractor.m.mask_layer
    ch_scores = [[] for _ in range(n_ch)]

    # Sub-sample indices evenly
    total = len(dataset)
    indices = list(np.linspace(0, total - 1, n_probe, dtype=int))
    probe_subset = torch.utils.data.Subset(dataset, indices)
    probe_loader = DataLoader(probe_subset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    logger.info(f"Probing {n_probe} frames to detect instrument channel ...")

    for batch in probe_loader:
        batch = to_device(batch)
        try:
            internals = extractor.extract(batch)
            soft_mask = internals["soft_mask"][0, 0]  # [C, H, W]
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

    # Model
    model = build_model(cfg, output_dir=str(out_dir))
    model = load_checkpoint(model, args.ckpt)

    # Extractors / scorers
    extractor  = RCFFeatureExtractor(model)
    maa_scorer = MAAScorer(arch=args.dino_arch,
                           patch_size=args.dino_patch_size,
                           tau=args.dino_tau)
    flow_scorer = FlowScorer(model) if args.use_flow else None

    if not args.use_flow:
        logger.info("Flow scoring DISABLED — using MAA only")
    else:
        logger.info(f"Flow scoring ENABLED  — alpha={args.alpha} "
                    f"(MAA:{args.alpha:.2f}  flow:{1-args.alpha:.2f})")

    # Dataset
    dataset = build_dataset(cfg, use_flow=args.use_flow)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    logger.info(f"Dataset: {len(dataset)} frames  "
                f"({'with' if args.use_flow else 'without'} flow)")

    # ── Step 1: detect instrument channel globally ───────────────────────────
    if args.object_channel is not None:
        obj_ch = args.object_channel
        logger.info(f"Using fixed object_channel={obj_ch} from CLI")
    else:
        obj_ch = detect_object_channel(
            extractor, maa_scorer, dataset,
            n_probe=args.channel_probe_frames,
            probe_th=args.eval_pos_th)
    logger.info(f"Instrument channel: {obj_ch}")

    ensemble = CombinedEnsemble(
        maa_scorer=maa_scorer,
        flow_scorer=flow_scorer,
        thresholds=args.thresholds,
        temperature=args.maa_temperature,
        alpha=args.alpha,
        object_channel=obj_ch,   # fixed channel passed in
    )

    # ── Step 2: inference on all frames ─────────────────────────────────────
    log_lines = [f"instrument_channel: {obj_ch}"]

    for batch in tqdm(loader, desc="MAA+Flow ensemble"):
        batch = to_device(batch)
        seq   = batch["seq_names"][0]
        fidx  = int(batch.get("frame_ind_start", [0])[0])

        internals = extractor.extract(batch)
        soft_mask = internals["soft_mask"][0, 0]   # [C, H, W]

        gt_fw = (torch.stack(batch["gt_fw_flows"], dim=1)
                 if args.use_flow and "gt_fw_flows" in batch else None)
        gt_bw = (torch.stack(batch["gt_bw_flows"], dim=1)
                 if args.use_flow and "gt_bw_flows" in batch else None)

        img_01 = get_img_01(batch)

        final_soft, best_th, score_log = ensemble.combine(
            soft_mask, img_01, obj_ch, internals, gt_fw, gt_bw)

        binary = (final_soft > args.eval_pos_th).cpu().numpy().astype(np.float32)
        fname  = f"{fidx:05d}"
        save_pair(img_01, binary, out_dir / "results", seq, fname)

        detail = " | ".join(
            f"th={s['th']:.2f} maa={s['maa']:.3f} "
            f"flow={s['flow']:.3f} comb={s['combined']:.3f}"
            for s in score_log)
        log_lines.append(
            f"{seq}/{fname}  best_th={best_th:.2f}  [{detail}]")

    (out_dir / "scores.txt").write_text("\n".join(log_lines))
    logger.info(f"Done. Results in {out_dir / 'results'} | Log: {out_dir / 'scores.txt'}")


# ════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--output",  required=True)

    p.add_argument("--use_flow", action="store_true",
                   help="Enable flow scoring (requires flow files on disk)")
    p.add_argument("--flow_suffix", default="_NewCT",
                   help="Flow file suffix (overrides config)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight of MAA score (1-alpha = weight of flow score)")

    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    p.add_argument("--maa_temperature",      type=float, default=10.0)
    p.add_argument("--eval_pos_th",          type=float, default=0.35)
    p.add_argument("--object_channel",       type=int,   default=None,
                   help="Fix instrument channel (skip auto-detection if set)")
    p.add_argument("--channel_probe_frames", type=int,   default=10,
                   help="Number of frames to sample for channel auto-detection")

    p.add_argument("--dino_arch",       default="vit_small")
    p.add_argument("--dino_patch_size", type=int, default=8)
    p.add_argument("--dino_tau",        type=float, default=0.2)
    p.add_argument("--workers",         type=int, default=4)

    run(p.parse_args())

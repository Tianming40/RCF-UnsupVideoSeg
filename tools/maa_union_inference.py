#!/usr/bin/env python3
"""
Flow Union Inference — unsupervised multi-channel instrument mask output.

No annotations required. Uses optical flow reconstruction quality (warp loss)
to identify instrument channels, which works better than MAA for distinguishing
instruments from semantically-coherent soft tissue.

Workflow:
  1. Probe N frames (with optical flow) → score each channel by warp loss
     (lower warp loss = channel better explains the optical flow = instrument)
  2. Greedy union: start with best single channel, add channels that
     improve per-sequence flow reconstruction (if applicable)
  3. Aggregate across probe frames per sequence → lock channel set
  4. Full inference: element-wise max over selected channels → threshold → save

Usage:
  python tools/maa_union_inference.py \\
    --config  configs/instrument/test_cmc_grasp10_maskonly.yaml \\
    --ckpt    saved/phase1_YYMMDD/epoch=52-step=8957.ckpt \\
    --output  saved/flow_union_output \\
    --split   ImageSets/trainval.txt \\
    --flow_suffix _NewCT \\
    --n_probe 20 \\
    --union_rel_th 0.3
"""

import argparse
import sys
import logging
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "SemanticConstraintsAndMAA"))

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
logging.getLogger("PIL").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (reused from maa_inference_ensemble_areaNorm.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, output_dir: str = "/tmp") -> RCFModel:
    import argparse as _ap, copy
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=True, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    # Enable mask resize so variable-size images (e.g. CMC) work with fixed mask_size
    cfg["model_kwargs"]["allow_mask_resize"] = True
    return RCFModel(args=fake_args, **copy.deepcopy(cfg["model_kwargs"]))


def load_checkpoint(model: RCFModel, ckpt_path: str) -> RCFModel:
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class RCFFeatureExtractor:
    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def extract(self, batch: dict) -> dict:
        imgs = torch.stack(batch["imgs"], dim=1).to(DEVICE)
        B, im_num, C3, H, W = imgs.shape
        img_flat = imgs.view(B * im_num, C3, H, W)
        all_feat = self.m.extract_feat(img_flat, self.m.backbone2)
        mask_pre = self.m._decode_head_forward(all_feat, self.m.decode_head2)
        _, _, fh, fw = mask_pre.shape
        soft_mask = mask_pre.view(B, im_num, self.m.mask_layer, fh, fw)
        soft_mask = F.softmax(soft_mask, dim=2)
        # Residuals needed for FlowScorer
        res_fw = res_bw = None
        if self.m.separate_residual:
            res_fw, res_bw = self.m.pred_separate_residual(all_feat, B, im_num)
        return dict(imgs=imgs, soft_mask=soft_mask,
                    res_fw=res_fw, res_bw=res_bw, B=B, im_num=im_num)


# ─────────────────────────────────────────────────────────────────────────────
# Flow scorer — picks the channel that best reconstructs optical flow
# (lower warp loss = more rigid/predictable motion = more likely instrument)
# ─────────────────────────────────────────────────────────────────────────────

class FlowScorer:
    def __init__(self, model: RCFModel):
        self.m = model

    @torch.no_grad()
    def score(self, internals: dict,
              gt_fw_flows: torch.Tensor,
              gt_bw_flows: torch.Tensor,
              candidate_channels: list,
              probe_th: float = 0.35) -> float:
        """
        Returns -warp_loss for the union of candidate_channels (higher = better).
        Only the candidate channels contribute to flow aggregation; others are zeroed.
        """
        B      = internals["B"]
        im_num = internals["im_num"]
        imgs   = internals["imgs"]

        # Build candidate mask: binarise the candidate channels at probe_th,
        # keep original soft values for non-candidate channels, then renormalise.
        # Adding a small floor (1e-6) prevents NaN from spatial normalisation
        # when a channel has near-zero probability everywhere.
        soft = internals["soft_mask"]            # [B, im_num, C, fh, fw]
        mask_size = tuple(self.m.mask_size)
        # Resize soft_mask to mask_size so it matches the flow (which is also
        # resized to mask_size in resize_flow).  Backbone output size may differ
        # from mask_size for variable-resolution datasets (e.g. CMC).
        if soft.shape[-2:] != torch.Size(list(mask_size)):
            B_, T_, C_, H_, W_ = soft.shape
            soft = F.interpolate(
                soft.view(B_ * T_ * C_, 1, H_, W_),
                mask_size, mode='bilinear', align_corners=False,
            ).view(B_, T_, C_, *mask_size)
        all_pred_mask = soft.clone()
        for ch in candidate_channels:
            all_pred_mask[:, :, ch] = (soft[:, :, ch] > probe_th).float()
        all_pred_mask = all_pred_mask + 1e-6     # floor to avoid 0/0
        total = all_pred_mask.sum(dim=2, keepdim=True)
        all_pred_mask = all_pred_mask / total    # renormalise across channels
        flow_num  = gt_fw_flows.shape[1]

        def resize_flow(f):
            if f.shape[-1] == 2:
                f = f.permute(0, 1, 4, 2, 3).contiguous()
            H_, W_ = f.shape[-2], f.shape[-1]
            return self.m.resize(
                f.view(B * flow_num, 2, H_, W_), mask_size
            ).view(B, flow_num, 2, *mask_size)

        fw_r = resize_flow(gt_fw_flows)
        bw_r = resize_flow(gt_bw_flows)

        # mask resize is handled by the model itself (allow_mask_resize=True in build_model)

        _, loss_flow = self.m.decode_head(
            imgs, all_pred_mask, fw_r, bw_r,
            internals["res_fw"], internals["res_bw"]
        )
        seg = loss_flow.get("seg",
              loss_flow.get("seg_fw", torch.tensor(0.)) +
              loss_flow.get("seg_bw", torch.tensor(0.)))
        val = seg.item() if isinstance(seg, torch.Tensor) else float(seg)
        return -val  # higher = better (lower warp loss)


# ─────────────────────────────────────────────────────────────────────────────
# MAA scorer
# ─────────────────────────────────────────────────────────────────────────────

class MAAScorer:
    def __init__(self, arch="vit_small", patch_size=8, tau=0.2, eps=1e-5):
        self.patch_size = patch_size
        self.tau, self.eps = tau, eps
        logger.info(f"Loading DINO {arch} p={patch_size}")
        self.dino = get_dino_model(arch, patch_size, device=DEVICE)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()
        self._feat_out = {}
        def hook(m, i, o): self._feat_out["qkv"] = o
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
        feat = qkv[1].transpose(1, 2).reshape(nb, nt, -1)
        return feat, h_feat, w_feat

    @torch.no_grad()
    def score(self, img_01: torch.Tensor, binary_mask: torch.Tensor) -> float:
        feats, h_feat, w_feat = self._get_feats(img_01)
        mask_feat = F.interpolate(binary_mask[None, None].float(),
                                  (h_feat, w_feat), mode="nearest")[0, 0]
        ncut = soft_ncut_value(feats, mask_feat, self.tau, self.eps)
        return float(-ncut)  # higher = better


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class _DummyAnn:
    def __call__(self, data):
        if 'ann' not in data:
            data['ann'] = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        return data


def _get_transform(cfg):
    import dataset as _ds, argparse as _ap
    fake_args = _ap.Namespace(
        train_transform_kwargs=cfg.get("train_transform_kwargs", {"strong_aug": True}),
        test_transform_kwargs=cfg.get("test_transform_kwargs", {"strong_aug": False}),
    )
    return _ds.get_transform(fake_args, training=False)


def build_dataset(cfg: dict, split_override: str = None) -> VideoDataset:
    """Inference dataset: single frame, no flow."""
    test_kw   = cfg.get("test_dataset_kwargs", {})
    data_path = cfg.get("data_path")
    split     = split_override or test_kw.get("split", "val.txt")
    return VideoDataset(
        root=data_path, split=split,
        training=False, frame_num=1, load_flow=False,
        zero_ann=True, transform=_get_transform(cfg),
    )


def build_probe_dataset(cfg: dict, split_override: str = None,
                        flow_suffix: str = "_NewCT") -> VideoDataset:
    """Probe dataset: two frames + optical flow for flow scoring."""
    train_kw  = cfg.get("train_dataset_kwargs", {})
    test_kw   = cfg.get("test_dataset_kwargs", {})
    data_path = cfg.get("data_path")
    split     = split_override or test_kw.get("split", "val.txt")

    class _WithAnn:
        def __init__(self, t): self._t = t
        def __call__(self, data):
            if 'ann' not in data:
                data['ann'] = Image.fromarray(np.zeros((1,1,3), dtype=np.uint8))
            return self._t(data)

    return VideoDataset(
        root=data_path, split=split,
        training=True,   # training=True so that flow is loaded
        frame_num=2, load_flow=True,
        flow_suffix=flow_suffix,
        flow_suffix2=train_kw.get("flow_suffix2", flow_suffix),
        flow_suffix3=train_kw.get("flow_suffix3", flow_suffix),
        zero_ann=True,
        transform=_WithAnn(_get_transform(cfg)),
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


# ─────────────────────────────────────────────────────────────────────────────
# Core: probe → select channels
# ─────────────────────────────────────────────────────────────────────────────

def detect_instrument_channels(extractor: RCFFeatureExtractor,
                                flow_scorer: FlowScorer,
                                probe_dataset: VideoDataset,
                                n_probe: int = 20,
                                probe_th: float = 0.35,
                                union_rel_th: float = 0.3,
                                flow_improve_th: float = 0.01,
                                min_new_coverage: float = 0.02) -> tuple:
    """
    Flow-based greedy channel detection. For each probe frame:
      1. Score all single channels by warp loss (lower = better)
      2. Start with best single channel (lowest warp loss)
      3. Greedily add channels that improve warp reconstruction AND add new coverage
      4. Record which channels were in the best union for this frame
    Aggregate across frames: include channels present in >= union_rel_th of frames.

    Uses optical flow → works for instruments (rigid motion) not soft tissue.
    """
    n_ch = extractor.m.mask_layer
    ch_freq = [0] * n_ch
    n_valid_frames = 0

    total   = len(probe_dataset)
    indices = list(np.linspace(0, total - 1, min(n_probe, total), dtype=int))
    probe_loader = DataLoader(Subset(probe_dataset, indices),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    logger.info(f"Probing {len(indices)} frames with flow scoring ...")

    for batch in tqdm(probe_loader, desc="Probing channels (flow)"):
        batch = to_device(batch)
        try:
            internals  = extractor.extract(batch)
            soft_mask  = internals["soft_mask"][0, 0]   # [C, fh, fw]
            gt_fw = torch.stack(batch["gt_fw_flows"], dim=1)
            gt_bw = torch.stack(batch["gt_bw_flows"], dim=1)

            # Step 1: score each single channel by flow reconstruction
            single_scores = []
            for ch in range(n_ch):
                s = flow_scorer.score(internals, gt_fw, gt_bw, [ch], probe_th)
                single_scores.append(s)
            logger.info(f"  single flow scores: {[f'{s:.4f}' for s in single_scores]}")

            # Step 2: start with best single channel (highest = lowest warp loss)
            best_ch    = int(np.argmax(single_scores))
            active     = [best_ch]
            best_score = single_scores[best_ch]

            # Step 3: greedily add channels that improve flow reconstruction
            current_union = (soft_mask[best_ch] > probe_th).float()
            remaining = sorted([k for k in range(n_ch) if k != best_ch],
                               key=lambda k: single_scores[k], reverse=True)
            for ch in remaining:
                new_ch_mask = (soft_mask[ch] > probe_th).float()
                new_pixels  = (new_ch_mask * (1 - current_union)).mean().item()
                if new_pixels < min_new_coverage:
                    continue
                s = flow_scorer.score(internals, gt_fw, gt_bw,
                                      active + [ch], probe_th)
                if s - best_score > flow_improve_th:
                    active.append(ch)
                    best_score    = s
                    current_union = torch.max(current_union, new_ch_mask)

            for ch in active:
                ch_freq[ch] += 1
            n_valid_frames += 1

        except Exception as e:
            logger.warning(f"Probe frame skipped: {e}")

    if n_valid_frames == 0:
        logger.warning("No valid probe frames — falling back to channel 0")
        return [0], ch_freq

    freq_th = union_rel_th * n_valid_frames
    valid_channels = [k for k in range(n_ch) if ch_freq[k] >= freq_th]
    if not valid_channels:
        valid_channels = [int(np.argmax(ch_freq))]

    logger.info(f"Channel participation ({n_valid_frames} probe frames):")
    for ch, freq in enumerate(ch_freq):
        pct = freq / n_valid_frames * 100
        marker = " ← selected" if ch in valid_channels else ""
        logger.info(f"  channel {ch}: {freq}/{n_valid_frames} ({pct:.0f}%){marker}")
    logger.info(f"Union channels: {valid_channels}  (freq_th={union_rel_th*100:.0f}%)")

    return valid_channels, ch_freq


# ─────────────────────────────────────────────────────────────────────────────
# Core: full inference with union mask
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model: RCFModel,
                  dataset: VideoDataset,
                  channels: list,
                  out_dir: Path,
                  eval_pos_th: float = 0.35,
                  workers: int = 4):
    """
    Run inference on all frames. For each frame saves:
      saved_eval/<seq_name>_<frame>.jpg  — grid: original + 5 channels + union mask
      masks/<seq>/<frame>.png            — binary union mask (full resolution)

    Visualization reuses model.forward_eval() + model.finalize_eval_save(),
    exactly matching the main.py eval format, with union mask appended as last row.
    """
    mask_dir = out_dir / "masks"
    # Point model's save_dir_eval to out_dir so forward_eval saves there
    model.save_dir_eval = str(out_dir / "saved_eval")
    Path(model.save_dir_eval).mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=True)

    logger.info(f"Running inference on {len(dataset)} frames "
                f"| union channels: {channels} | output: {out_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = to_device(batch)
            seq   = batch["seq_names"][0]
            fidx  = int(batch.get("frame_ind_start", [0])[0])

            # Stack imgs: [B, im_num, 3, H, W]
            imgs = torch.stack(batch["imgs"], dim=1)

            # forward_eval builds tosave (original + 5 channels) and stores in
            # model._pending_eval_viz; returns pred_masks [B, C, fh, fw]
            pred_masks = model.forward_eval(
                imgs, batch["seq_ids"], batch["seq_names"], batch["paths"])

            # Compute union mask (soft, then binary for finalize row)
            soft = pred_masks[0]   # [C, fh, fw]
            union_prob = soft[channels[0]].clone()
            for ch in channels[1:]:
                union_prob = torch.max(union_prob, soft[ch])

            # finalize_eval_save appends union row and writes the jpg
            union_np = (union_prob > eval_pos_th).cpu().numpy().astype('float32')
            model.finalize_eval_save([union_np])

            # Also save binary mask PNG at full resolution
            img_h, img_w = batch["imgs"][0].shape[-2:]
            union_full = F.interpolate(
                union_prob[None, None].float(), (img_h, img_w),
                mode="bilinear", align_corners=False)[0, 0]
            binary = (union_full > eval_pos_th).cpu().numpy().astype(np.uint8) * 255
            seq_dir = mask_dir / seq
            seq_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary).save(str(seq_dir / f"{fidx:05d}.png"))

    logger.info(f"Vis   → {out_dir / 'saved_eval'}")
    logger.info(f"Masks → {mask_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    cfg     = load_config(args.config)
    out_dir = Path(args.output)

    model     = build_model(cfg, output_dir=str(out_dir))
    model     = load_checkpoint(model, args.ckpt)
    extractor = RCFFeatureExtractor(model)

    infer_dataset = build_dataset(cfg, split_override=args.split)

    # ── Phase 1: probe to determine channels ────────────────────────────────
    if args.channels is not None:
        channels = args.channels
        logger.info(f"Using manually specified channels: {channels}")
    else:
        flow_scorer   = FlowScorer(model)
        probe_dataset = build_probe_dataset(
            cfg, split_override=args.split,
            flow_suffix=args.flow_suffix)

        channels, ch_freq = detect_instrument_channels(
            extractor, flow_scorer, probe_dataset,
            n_probe=args.n_probe,
            probe_th=args.probe_th,
            union_rel_th=args.union_rel_th,
        )

    # ── Phase 2: full inference ──────────────────────────────────────────────
    run_inference(
        model, infer_dataset, channels,
        out_dir=out_dir,
        eval_pos_th=args.eval_pos_th,
        workers=args.workers,
    )

    # Save run config for reproducibility
    summary = {
        "ckpt":          args.ckpt,
        "split":         args.split or cfg.get("test_dataset_kwargs", {}).get("split"),
        "channels":      channels,
        "union_rel_th":  args.union_rel_th,
        "probe_th":      args.probe_th,
        "eval_pos_th":   args.eval_pos_th,
        "n_probe":       args.n_probe,
    }
    (out_dir / "run_config.yaml").write_text(yaml.dump(summary))
    logger.info(f"Done. Masks: {out_dir / 'masks'} | Config: {out_dir / 'run_config.yaml'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Flow-based union channel inference (no annotations required)")
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--split",   default=None)

    # Flow
    p.add_argument("--flow_suffix", default="_NewCT",
                   help="Flow suffix to use for probe dataset (e.g. _NewCT)")

    # Channel selection
    p.add_argument("--n_probe",       type=int,   default=20)
    p.add_argument("--probe_th",      type=float, default=0.35)
    p.add_argument("--union_rel_th",  type=float, default=0.3,
                   help="Include channel if it wins >= this fraction of probe frames")
    p.add_argument("--channels",      type=int,   nargs="+", default=None,
                   help="Manually specify channels (skips flow probing)")

    # Inference
    p.add_argument("--eval_pos_th",   type=float, default=0.35)
    p.add_argument("--workers",       type=int,   default=4)

    run(p.parse_args())

"""
main_crf_posttrain.py  –  MAA-guided CRF post-training.

Two stages:
  1. Load a trained checkpoint, probe the val set with DINO NCut (MAA) to
     auto-detect the object channel.
  2. Short post-training with CRF pseudo-label loss active from epoch 0.
     Training infrastructure is reused directly from main.py / main_v2.py.

Usage:
  python main_crf_posttrain.py configs/instrument/rcf_cmc_crf_posttrain_v2b.yaml \\
      --ckpt saved/saved_cmc_all_finetune_v2b_0528_132020/epoch=17-step=6498.ckpt \\
      [--output_dir saved/crf_posttrain_run1] \\
      [--crf_epochs 10] [--w_crf 0.5] [--crf_lr 2e-5] \\
      [--probe_frames 20] [--probe_th 0.35] \\
      [--crf_iters 10] [--crf_sxy 60.] [--crf_srgb 5.]
"""

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# ── V2 monkey-patch (same as main_v2.py) ───────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import dataset
import models
import utils
import wandb

# Reuse Model and CustomProgressBar from main.py directly
import main as _main_mod
from main import Model, CustomProgressBar

# soft_ncut_value from tools/SemanticConstraintsAndMAA/maa.py
sys.path.insert(0, str(Path(__file__).parent / "tools" / "SemanticConstraintsAndMAA"))
from maa import soft_ncut_value

logger = utils.get_logger()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1: MAA channel detection  (new code, only part not in main.py)
# ════════════════════════════════════════════════════════════════════════════════

class _MAAScorer:
    """DINO NCut scorer, works on arbitrary resolution via padding."""

    def __init__(self, arch="vit_small", patch_size=8, which_features="k",
                 tau=0.2, eps=1e-5):
        from models.dino_vit import get_dino_model
        self.ps       = patch_size
        self.tau      = tau
        self.eps      = eps
        self.feat_key = {"k": 1, "q": 0, "v": 2}[which_features]

        self.dino = get_dino_model(arch, patch_size, device=DEVICE)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

        self._feat_out = {}
        def _hook(m, i, o): self._feat_out["qkv"] = o
        self.dino._modules["blocks"][-1]._modules["attn"] \
            ._modules["qkv"].register_forward_hook(_hook)

        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[None, :, None, None]
        self._mean, self._std = mean, std

    @torch.no_grad()
    def score(self, img_01: torch.Tensor, binary_mask: torch.Tensor) -> float:
        """img_01: [1,3,H,W] 0-1 · binary_mask: [H,W] 0/1 → MAA (higher = better)"""
        x = (img_01 - self._mean) / self._std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.ps, 0, (-h) % self.ps))
        h_f, w_f = x.shape[-2] // self.ps, x.shape[-1] // self.ps

        self._feat_out = {}
        att = self.dino.get_last_selfattention(x)
        nb, nh, nt = att.shape[:3]
        qkv  = (self._feat_out["qkv"]
                .reshape(nb, nt, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4))
        feat = qkv[self.feat_key].transpose(1, 2).reshape(nb, nt, -1)

        mask_f = F.interpolate(binary_mask[None, None].float(),
                               (h_f, w_f), mode="nearest")[0, 0]
        return float(-soft_ncut_value(feat, mask_f, self.tau, self.eps))


def detect_object_channel(rcf_model, probe_dataset,
                           n_probe=20, probe_th=0.35,
                           maa_scorer=None) -> int:
    """Sample n_probe frames, score every channel by MAA, return best channel."""
    if maa_scorer is None:
        maa_scorer = _MAAScorer()

    n_ch      = rcf_model.mask_layer
    ch_scores = [[] for _ in range(n_ch)]
    total     = len(probe_dataset)
    indices   = list(np.linspace(0, total - 1, min(n_probe, total), dtype=int))
    loader    = DataLoader(torch.utils.data.Subset(probe_dataset, indices),
                           batch_size=1, shuffle=False, num_workers=0)

    logger.info(f"MAA probe: {len(indices)} frames, {n_ch} channels ...")
    for batch in loader:
        imgs = torch.stack([t.to(DEVICE) for t in batch["imgs"]], dim=1)
        B, im_num, C3, H, W = imgs.shape
        try:
            with torch.no_grad():
                feat = rcf_model.extract_feat(imgs.view(B * im_num, C3, H, W),
                                              rcf_model.backbone2)
                pre  = rcf_model._decode_head_forward(feat, rcf_model.decode_head2)
                _, _, fh, fw = pre.shape
                soft = F.softmax(pre.view(B, im_num, n_ch, fh, fw), dim=2)[0, 0]

            img_01 = ((imgs[0, 0] + 2.0) / 4.0).clamp(0, 1)[None]  # [1,3,H,W]
            for ch in range(n_ch):
                binary = (soft[ch] > probe_th).float()
                frac   = binary.mean().item()
                if frac < 0.005 or frac > 0.995:
                    continue
                binary_img = F.interpolate(binary[None, None], (H, W),
                                           mode="nearest")[0, 0]
                ch_scores[ch].append(maa_scorer.score(img_01, binary_img))
        except Exception as e:
            logger.warning(f"  skipped frame: {e}")

    avg  = [float(np.mean(v)) if v else -1e9 for v in ch_scores]
    best = int(np.argmax(avg))
    for ch, s in enumerate(avg):
        logger.info(f"  ch {ch}: MAA={s:.4f}{'  ← best' if ch == best else ''}")
    return best


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2: CRF fine-tuning  (mirrors main.py's main(), reuses Model / CustomProgressBar)
# ════════════════════════════════════════════════════════════════════════════════

def run(cli_args):
    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank <= 0:
        utils.set_loglevel(debug=True)
    else:
        utils.set_loglevel(debug=False)

    logger.info(f"Loading config: {cli_args.config}")
    args = utils.load_args(cli_args.config, cli_opts=[])

    # ── Stage 1: MAA channel detection ────────────────────────────────────────
    logger.info("=== Stage 1: MAA channel detection ===")

    import argparse as _ap
    fake = _ap.Namespace(
        train_transform_kwargs=getattr(args, "train_transform_kwargs", {}),
        test_transform_kwargs=getattr(args, "test_transform_kwargs", {"strong_aug": False}),
    )
    test_kw   = getattr(args, "test_dataset_kwargs", {})
    data_path = getattr(args, "test_data_path", None) or args.data_path

    probe_dataset = dataset.VideoDataset(
        root=data_path,
        split=test_kw.get("split", "val.txt"),
        training=False, frame_num=1, load_flow=False,
        zero_ann=test_kw.get("zero_ann", True),
        transform=dataset.get_transform(fake, training=False),
    )
    logger.info(f"Probe dataset: {len(probe_dataset)} frames from {data_path}")

    os.makedirs("/tmp/crf_ft_probe", exist_ok=True)
    probe_model = models.__dict__[args.model_cls](
        args=_ap.Namespace(checkpoints_dir="/tmp/crf_ft_probe",
                           eval_save=False, eval_export=False,
                           export_all_seg=False, eval_pos_th=0.35,
                           object_channel=None, log_interval=9999),
        **copy.deepcopy(args.model_kwargs),
    )
    ckpt = torch.load(cli_args.ckpt, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)
    sd   = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    logger.info(f"Probe model mismatches: {probe_model.load_state_dict(sd, strict=False)}")
    probe_model.eval().to(DEVICE)

    obj_ch = detect_object_channel(probe_model, probe_dataset,
                                   n_probe=cli_args.probe_frames,
                                   probe_th=cli_args.probe_th,
                                   maa_scorer=_MAAScorer())
    logger.info(f"=== Object channel: {obj_ch} ===")
    del probe_model; torch.cuda.empty_cache()

    # ── Stage 2: inject CRF params, then train exactly like main.py ───────────
    logger.info("=== Stage 2: CRF fine-tuning ===")

    args.model_kwargs.update(dict(
        w_crf=cli_args.w_crf,
        crf_pos_weight=cli_args.crf_pos_weight,
        crf_neg_weight=cli_args.crf_neg_weight,
        crf_mask_pos_th=-1.,
        crf_use_ema=False,
        crf_head=dict(type="CRFHead", srgb=cli_args.crf_srgb, sxy=cli_args.crf_sxy,
                      scomp=5., refine_iters=cli_args.crf_iters, crf_scale=0.7),
    ))
    args.object_channel  = obj_ch
    args.epochs          = cli_args.crf_epochs
    args.learning_rate   = cli_args.crf_lr
    args.checkpoints_dir = cli_args.output_dir or (args.checkpoints_dir + "_crf_ft")
    args.pretrained_model               = cli_args.ckpt
    args.allow_overwriting_checkpoints_dir = True
    args.config_path                    = cli_args.config
    args.test                           = False
    args.rank                           = rank
    args.multi_gpu                      = rank > -1
    args.resume_from_checkpoint         = None

    # Update main.py's module-level exp_name so CustomProgressBar picks it up
    _main_mod.exp_name = (args.checkpoints_dir.split("/")[-1]
                          + "_" + datetime.now().strftime("%y%m%d_%H%M%S"))

    # Logger — mirrors main.py exactly
    if args.disable_wandb:
        wandb_logger = pl.loggers.TensorBoardLogger(
            save_dir=args.checkpoints_dir, name="tensorboard_logs",
            version=None, default_hp_metric=True)
    else:
        wandb_logger = pl.loggers.WandbLogger(
            project="RCF", mode=None, name=_main_mod.exp_name,
            settings=wandb.Settings(start_method="thread"))

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoints_dir, save_on_train_epoch_end=False,
        every_n_epochs=1, monitor="val_miou_frame_avg",
        save_top_k=5, save_last=True, mode="max", auto_insert_metric_name=True)

    trainer_cfg = {
        "logger":              wandb_logger,
        "max_epochs":          getattr(args, "override_max_epochs", args.epochs),
        "accelerator":         "gpu",
        "replace_sampler_ddp": False,
        "callbacks":           [CustomProgressBar(), checkpoint_cb],
        **args.trainer_kwargs,
    }
    if args.multi_gpu:
        trainer_cfg["strategy"] = "ddp_find_unused_parameters_false"
    else:
        trainer_cfg["devices"] = 1

    trainer = pl.Trainer(**trainer_cfg, default_root_dir=args.checkpoints_dir)
    model   = Model(args, trainer)

    trainer.fit(model=model, ckpt_path=None)
    args.saved_eval_dir_name = "saved_eval_test"
    args.eval_pos_th = -1
    trainer.test(model=model, ckpt_path="best")


def main():
    p = argparse.ArgumentParser(description="MAA-guided CRF fine-tuning.")
    p.add_argument("config",  type=str,
                   help="Base config yaml, e.g. configs/instrument/rcf_cmc_all_finetune_v2b.yaml")
    p.add_argument("--ckpt",  type=str, required=True,
                   help="Checkpoint for MAA probing and as the starting weights")
    p.add_argument("--output_dir",    type=str,   default=None)
    p.add_argument("--probe_frames",  type=int,   default=20)
    p.add_argument("--probe_th",      type=float, default=0.35)
    p.add_argument("--crf_epochs",    type=int,   default=10)
    p.add_argument("--crf_lr",        type=float, default=2e-5)
    p.add_argument("--w_crf",         type=float, default=0.5)
    p.add_argument("--crf_pos_weight",type=float, default=2.0)
    p.add_argument("--crf_neg_weight",type=float, default=1.0)
    p.add_argument("--crf_srgb",      type=float, default=5.)
    p.add_argument("--crf_sxy",       type=float, default=60.)
    p.add_argument("--crf_iters",     type=int,   default=10)
    run(p.parse_args())


if __name__ == "__main__":
    main()

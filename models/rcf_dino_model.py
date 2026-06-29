"""
rcf_dino_model.py — RCFDinoModel (Method B: DINO-guided Phase 1)

Adds a frozen DINO ViT-small as an auxiliary teacher to RCFModel.

Loss added:
  L_dino  (soft K-means in DINO feature space)
  For each channel c: penalise within-channel cosine-distance to the
  mask-weighted centroid in DINO feature space.

  L_dino = (1/C) Σ_c  Σ_pixel [ M̂_c(p) · (1 - cos(feat_p, μ_c)) ]
                                / Σ_pixel M̂_c(p)

  Where μ_c is the mask-weighted average of L2-normalised DINO key features.

Why this helps:
  - DINO ViT features naturally cluster visually similar regions
    (instrument metal, pink soft-tissue, dark background)
  - Adding L_dino forces each channel to contain visually coherent pixels,
    breaking the initial symmetry much faster than warp loss alone
  - DINO is frozen → no gradient flows back into the ViT, only into FCNHead
  - This is channel-index-agnostic: whichever channel ends up with tissue,
    DINO will make its pixels visually consistent

Design notes:
  - Inherits RCFModel; does NOT modify any existing file
  - Override _decode_head_forward to capture mask logits (same pattern as
    RCFTissueModel)
  - DINO wrapped in _FrozenModule to guarantee eval mode under PL .train()
  - DINO is loaded once at init; weights never updated
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rcf_model import RCFModel
import utils

logger = utils.get_logger()


class _FrozenModule(nn.Module):
    """Thin wrapper that keeps the inner module permanently in eval mode,
    even when PyTorch Lightning calls .train() on the parent."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # Always stay in eval regardless of what PL does
        return super().train(False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _build_dino(arch: str, patch_size: int, checkpoint: Optional[str]) -> nn.Module:
    """
    Build a DINO ViT and load weights.

    Args:
        arch:        'vit_small' or 'vit_base'
        patch_size:  8 or 16
        checkpoint:  local .pth path, or None → torch.hub auto-download
                     (uses ~/.cache/torch/hub/checkpoints if already cached)
    """
    from models.dino_vit import vit_small, vit_base

    if arch == "vit_small":
        model = vit_small(patch_size=patch_size, num_classes=0)
        default_url = (
            f"dino/dino_deitsmall{patch_size}_300ep_pretrain/"
            f"dino_deitsmall{patch_size}_300ep_pretrain.pth"
        )
    elif arch == "vit_base":
        model = vit_base(patch_size=patch_size, num_classes=0)
        default_url = (
            f"dino/dino_vitbase{patch_size}_pretrain/"
            f"dino_vitbase{patch_size}_pretrain.pth"
        )
    else:
        raise ValueError(f"Unsupported DINO arch: {arch}")

    if checkpoint is not None:
        logger.info(f"[DINO] Loading from local checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        # Some checkpoints are full training dumps with nested keys
        if "teacher" in state_dict:
            state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict["teacher"].items()
                if k.startswith("backbone.")
            }
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
    else:
        logger.info("[DINO] Downloading weights via torch.hub …")
        state_dict = torch.hub.load_state_dict_from_url(  # type: ignore[arg-type]
            url="https://dl.fbaipublicfiles.com/" + default_url,
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    logger.info(f"[DINO] Loaded {arch} patch{patch_size}")
    return model


class RCFDinoModel(RCFModel):
    """
    RCFModel augmented with a DINO visual-consistency loss (L_dino).

    Extra __init__ keyword arguments (all optional with defaults):

      dino_arch        str    ViT variant: 'vit_small' | 'vit_base'
                              default: 'vit_small'
      dino_patch_size  int    ViT patch size: 8 | 16
                              default: 8
      dino_checkpoint  Optional[str]  local .pth path for DINO weights.
                              None → auto-download via torch.hub
                              default: None
      w_dino           float  loss weight for L_dino
                              default: 0.1
      dino_input_size  int    spatial size to resize frames before DINO
                              (must be divisible by dino_patch_size)
                              default: 128
    """

    def __init__(
        self,
        args,
        dino_arch: str = "vit_small",
        dino_patch_size: int = 8,
        dino_checkpoint: Optional[str] = None,
        w_dino: float = 0.1,
        dino_input_size: int = 128,
        dino_channels: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(args, **kwargs)

        self.w_dino = w_dino
        self.dino_input_size = dino_input_size
        self.dino_patch_size = dino_patch_size
        # None = apply to all channels; [1] = ch1 only, etc.
        self.dino_channels = dino_channels

        # Build and freeze DINO
        raw_dino = _build_dino(dino_arch, dino_patch_size, dino_checkpoint)
        self.dino = _FrozenModule(raw_dino)

        # Slot for mask logits captured during forward_train
        self._captured_mask_logits: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Capture mask logits (same pattern as RCFTissueModel)                #
    # ------------------------------------------------------------------ #
    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if flow_feat is not None:
            pred = decode_head.forward(x, flow_feat=flow_feat)
        else:
            pred = decode_head.forward(x)
        if self.training and decode_head is self.decode_head2:
            self._captured_mask_logits = pred          # [B*I, C, H, W]
        return pred

    # ------------------------------------------------------------------ #
    # DINO feature extraction                                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _extract_dino_feats(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level DINO key features.

        Args:
            imgs: [B, 3, H, W]  — training images, already ImageNet-normalised
                  (same normalisation used by DenseCL backbone, directly
                  compatible with DINO which uses the same mean/std)

        Returns:
            feats: [B, D, H_p, W_p]  — L2-normalised key features per patch
        """
        S = self.dino_input_size
        # Resize to DINO input size
        if imgs.shape[-2] != S or imgs.shape[-1] != S:
            imgs_r = F.interpolate(
                imgs, (S, S), mode="bilinear", align_corners=False
            )
        else:
            imgs_r = imgs

        # DINO forward: [B, N_patches+1, D]  (first token is CLS)
        out = self.dino(imgs_r)
        patch_feats = out[:, 1:]                        # [B, N_patches, D]

        H_p = W_p = S // self.dino_patch_size
        D = patch_feats.shape[-1]

        # Reshape to spatial and L2-normalise along feature dim
        feats_2d = patch_feats.view(
            patch_feats.shape[0], H_p, W_p, D
        ).permute(0, 3, 1, 2)                           # [B, D, H_p, W_p]
        return F.normalize(feats_2d, dim=1)

    # ------------------------------------------------------------------ #
    # DINO soft K-means consistency loss                                   #
    # ------------------------------------------------------------------ #
    def _dino_consistency_loss(
        self, masks: torch.Tensor, imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L_dino: soft K-means in frozen DINO feature space.

        For each channel c:
          1. Compute mask-weighted centroid μ_c of L2-normalised DINO features
          2. Penalise mask-weighted (1 - cosine_sim(patch, μ_c))

        This loss is channel-index-agnostic: it only asks that pixels within
        the same channel are visually similar, regardless of which channel
        corresponds to tissue or instrument.

        Args:
            masks: [B, C, H_m, W_m]  soft masks after softmax (values in [0,1])
            imgs:  [B, 3, H, W]      first-frame training images

        Returns:
            scalar loss
        """
        B, C, H_m, W_m = masks.shape
        channels = self.dino_channels if self.dino_channels is not None else list(range(C))

        # --- DINO features (no grad, frozen) ---
        feats = self._extract_dino_feats(imgs)          # [B, D, H_p, W_p]
        _, D, H_p, W_p = feats.shape

        # Resize masks to DINO patch resolution (keep gradient for backprop)
        if H_m != H_p or W_m != W_p:
            masks_p_grad = F.interpolate(
                masks, (H_p, W_p), mode="bilinear", align_corners=False
            )
        else:
            masks_p_grad = masks                        # [B, C, H_p, W_p]

        total = torch.tensor(0.0, device=masks.device)

        for c in channels:
            w_c = masks_p_grad[:, c]                    # [B, H_p, W_p]
            W_c = w_c.sum(dim=(1, 2))                   # [B]

            # Mask-weighted centroid: [B, D]
            centroid = (feats * w_c.unsqueeze(1)).sum(dim=(2, 3))
            centroid = centroid / (W_c.unsqueeze(1) + 1e-6)
            centroid = F.normalize(centroid, dim=1)     # [B, D]

            # Cosine similarity of each patch to centroid
            sim = (feats * centroid.view(B, D, 1, 1)).sum(dim=1)  # [B, H_p, W_p]

            # Mask-weighted (1 - cosine_sim): lower is better
            loss_c = (w_c * (1.0 - sim)).sum(dim=(1, 2)) / (W_c + 1e-6)
            total = total + loss_c.mean()

        return total / len(channels)

    # ------------------------------------------------------------------ #
    # Override forward_train to append L_dino                             #
    # ------------------------------------------------------------------ #
    def forward_train(self, imgs, seq_ids, seq_names, paths,
                      gt_fw_flows, gt_bw_flows, pl_masks):
        self._captured_mask_logits = None
        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks
        )

        if self.w_dino <= 0.0 or self._captured_mask_logits is None:
            return losses

        # Reconstruct soft masks from captured logits
        # _captured_mask_logits: [B*I, C, H, W]
        batch_im_num = self._captured_mask_logits.shape[0]
        C = self._captured_mask_logits.shape[1]
        batch_size = imgs.shape[0]
        im_num = batch_im_num // batch_size   # typically 2

        # Softmax along channel dim
        masks_all = F.softmax(
            self._captured_mask_logits.view(batch_size, im_num, C,
                                            self._captured_mask_logits.shape[2],
                                            self._captured_mask_logits.shape[3]),
            dim=2
        )                                                # [B, I, C, H, W]

        # Apply DINO loss on the first frame only (frame 0)
        masks_frame0 = masks_all[:, 0]                  # [B, C, H, W]
        imgs_frame0 = imgs[:, 0]                        # [B, 3, H, W]

        l_dino = self._dino_consistency_loss(masks_frame0, imgs_frame0)

        losses["loss_dino"] = l_dino
        losses["loss"] = losses["loss"] + self.w_dino * l_dino

        return losses

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

      use_dino_graph_fusion    bool   enable DinoGraphFusionHead (see
                                      models/dino_graph_fusion.py), discussed
                                      260730 -- fuses frozen-DINO graph-
                                      partitioning eigenvectors into backbone2's
                                      feat0 before decode_head2, via a small
                                      trainable, zero-init'd conv module.
                                      default: False (zero-behaviour-change for
                                      every existing config, incl. v102)
      dino_graph_input_size    int    resize frames to this square before DINO
                                      for the graph branch (independent of
                                      dino_input_size, which stays 128 for
                                      loss_dino). default: 384 (matches
                                      backbone2's own input res -- the
                                      configuration empirically validated in
                                      saved/edge_test_260729/spectral_test.jpg)
      dino_graph_grid_size     int    graph node grid GxG (default 32 -> 1024
                                      nodes, same as spectral_test.jpg)
      dino_graph_num_eigvecs   int    how many smallest-eigenvalue eigenvectors
                                      to use (default 10)
      dino_graph_proj_channels int    channel width of the small conv module's
                                      internal eigenvector projection (default 32)
      dino_graph_feat_channels int    channel count of backbone2's feat0
                                      (default 256, matches ResNet50 stage0)
      dino_graph_chunk_size    int    max frames processed through DINO+eigh
                                      per internal chunk (default 8) -- purely
                                      a peak-memory control (DINO-384's self-
                                      attention is O(tokens^2)=O(2304^2) per
                                      sample), no_grad throughout so chunking
                                      doesn't change results. Does NOT affect
                                      batch_size/topk training semantics --
                                      deliberately kept independent so a
                                      memory fix here never silently re-tunes
                                      topk's easy-sample survival ratio
                                      (discussed 260730).

      use_dino_graph_estep_fusion  bool  enable DinoGraphEStepFusion (see
                                      models/dino_graph_estep_fusion.py),
                                      discussed 260730 -- an alternative to
                                      use_dino_graph_fusion with NO new
                                      trainable weights: fuses P_CNN(z=k) with
                                      a graph-derived P_Graph(z=k) computed
                                      fresh each forward pass from the CNN's
                                      own current mask (self-referential
                                      centroid matching -- see that file's
                                      docstring). Independent of/compatible
                                      with use_dino_graph_fusion, though not
                                      intended to be enabled together in the
                                      configs used so far. default: False
                                      (zero-behaviour-change for every
                                      existing config, incl. v102 and v146).
      dino_graph_estep_*           see models/dino_graph_estep_fusion.py for
                                      input_size/grid_size/num_eigvecs/
                                      chunk_size (same meaning as the
                                      dino_graph_* params above, kept as a
                                      fully separate set so the two mechanisms
                                      never share config/state) plus
                                      temperature and alpha (blend exponent,
                                      alpha=0 -> exact no-op, alpha=1 -> the
                                      literal P_CNN*P_Graph fusion).

      use_dino_graph_fusion_deep   bool  discussed 260730: v146
                                      (use_dino_graph_fusion) fuses DINO-graph
                                      eigenvectors into feat0 (backbone2's
                                      stage-1 output, 256ch/H4, MultiScaleSegHead's
                                      LOCAL/fine-boundary-detail input). Eigenvectors
                                      encode GLOBAL semantic-partition structure,
                                      which is architecturally closer to what
                                      feat1/feat2/feat3 (deep, large-receptive-
                                      field, "what class is this roughly"
                                      features, fused together BEFORE feat0 is
                                      even consulted) are for -- not feat0's
                                      job. This reuses the exact same
                                      DinoGraphFusionHead class (see models/
                                      dino_graph_fusion.py, same zero-init
                                      no-op safety property) but targets
                                      all_feat[3] (feat3, the deepest/most-
                                      semantic scale) instead of all_feat[0].
                                      Separate attribute
                                      (dino_graph_fusion_deep_head), separate
                                      hook in rcf_model.py -- independent of/
                                      compatible with use_dino_graph_fusion,
                                      not intended to be enabled together in
                                      the configs used so far. default: False.
      dino_graph_deep_*            same meaning as dino_graph_* above
                                      (input_size/grid_size/num_eigvecs/
                                      proj_channels/chunk_size), but
                                      feat_channels defaults to 2048 (feat3's
                                      channel count, not feat0's 256) and this
                                      is a fully separate param/state set.

      use_dino_graph_attention_gate  bool  discussed 260801: an ALTERNATIVE
                                      fusion TYPE to use_dino_graph_fusion_deep
                                      (v149) at the same feat3 injection
                                      point -- instead of concatenating a
                                      projection of the eigenvectors into
                                      feat3 (where they're outnumbered 2048:32
                                      by feat3's own channels), predicts a
                                      single-channel spatial GATE from the
                                      eigenvectors and multiplicatively
                                      modulates feat3: F' = F*(1+tanh(gate)),
                                      gate's last conv zero-initialized so
                                      F'==F exactly at init. See models/
                                      dino_graph_attention_gate.py's docstring.
                                      Separate attribute
                                      (dino_graph_attention_gate_head),
                                      separate hook in rcf_model.py -- not
                                      intended to be enabled together with
                                      use_dino_graph_fusion_deep in the same
                                      config (both target feat3). default:
                                      False.
      dino_graph_gate_*            same meaning as dino_graph_deep_* above
                                      (input_size/grid_size/num_eigvecs/
                                      chunk_size) plus gate_hidden_channels
                                      (width of the small conv stack that
                                      predicts the gate logit, default 32) --
                                      fully separate param/state set from
                                      both dino_graph_* and dino_graph_deep_*.

      use_dino_graph_fusion_deep2  bool  discussed 260801: multi-scale
                                      injection -- reuses the exact same
                                      DinoGraphFusionHead class as
                                      dino_graph_fusion_deep_head (v149)
                                      but targets all_feat[2] (feat2,
                                      1024ch, H/8=48x48) as a SECOND,
                                      independent injection point, on top
                                      of (not instead of) feat3. Meant to be
                                      combined with use_dino_graph_fusion_deep
                                      in the same config -- orthogonal
                                      variable to fusion TYPE (concat vs
                                      attention-gate). Separate attribute
                                      (dino_graph_fusion_deep2_head),
                                      separate hook. default: False.
      dino_graph_deep2_*           same meaning as dino_graph_deep_* above,
                                      but feat_channels defaults to 1024
                                      (feat2's channel count) and this is a
                                      fully separate param/state set.

      use_dino_graph_decoder_prior bool  discussed 260801: a third distinct
                                      injection POINT (not just fusion type)
                                      -- instead of fusing into a backbone2
                                      feature (encoder-side, before
                                      decode_head2 runs, like v146/v149/
                                      v153), this builds an extraction-only
                                      DinoGraphEigvecExtractor (models/
                                      dino_graph_eigvec_extractor.py, NO
                                      trainable params) whose output is
                                      passed into decode_head2 as an extra
                                      dino_eigvecs kwarg -- only meaningful
                                      when decode_head2.type is
                                      MultiScaleSegHeadDecoderPrior (models/
                                      multi_scale_seg_head_decoder_prior.py),
                                      which owns its own trainable fusion
                                      module and injects at the DECODER stage
                                      (right after decode_head2's own multi-
                                      scale/ASPP fusion, before feat0 concat)
                                      -- see that file's docstring. For any
                                      other decode_head2 type this kwarg is
                                      simply unused (harmless). default: False.
      dino_prior_input_size/grid_size/num_eigvecs/chunk_size  extraction-side
                                      params (same meaning as dino_graph_*
                                      above) -- MUST match decode_head2's own
                                      dino_prior_num_eigvecs (the fusion
                                      module's expected input channel count),
                                      since those two are configured
                                      independently (model_kwargs top level
                                      vs decode_head2 sub-config).
    """

    def __init__(
        self,
        args,
        dino_arch: str = "vit_small",
        dino_patch_size: int = 8,
        dino_checkpoint: Optional[str] = None,
        w_dino: float = 0.1,
        w_dino_merge: float = 0.0,
        dino_input_size: int = 128,
        dino_channels: Optional[list] = None,
        use_dino_graph_fusion: bool = False,
        dino_graph_input_size: int = 384,
        dino_graph_grid_size: int = 32,
        dino_graph_num_eigvecs: int = 10,
        dino_graph_proj_channels: int = 32,
        dino_graph_feat_channels: int = 256,
        dino_graph_chunk_size: int = 8,
        use_dino_graph_estep_fusion: bool = False,
        dino_graph_estep_input_size: int = 384,
        dino_graph_estep_grid_size: int = 32,
        dino_graph_estep_num_eigvecs: int = 10,
        dino_graph_estep_chunk_size: int = 8,
        dino_graph_estep_temperature: float = 1.0,
        dino_graph_estep_alpha: float = 1.0,
        use_dino_graph_fusion_deep: bool = False,
        dino_graph_deep_input_size: int = 384,
        dino_graph_deep_grid_size: int = 32,
        dino_graph_deep_num_eigvecs: int = 10,
        dino_graph_deep_proj_channels: int = 32,
        dino_graph_deep_feat_channels: int = 2048,
        dino_graph_deep_chunk_size: int = 8,
        use_dino_graph_attention_gate: bool = False,
        dino_graph_gate_input_size: int = 384,
        dino_graph_gate_grid_size: int = 32,
        dino_graph_gate_num_eigvecs: int = 10,
        dino_graph_gate_hidden_channels: int = 32,
        dino_graph_gate_feat_channels: int = 2048,
        dino_graph_gate_chunk_size: int = 8,
        use_dino_graph_fusion_deep2: bool = False,
        dino_graph_deep2_input_size: int = 384,
        dino_graph_deep2_grid_size: int = 32,
        dino_graph_deep2_num_eigvecs: int = 10,
        dino_graph_deep2_proj_channels: int = 32,
        dino_graph_deep2_feat_channels: int = 1024,
        dino_graph_deep2_chunk_size: int = 8,
        use_dino_graph_decoder_prior: bool = False,
        dino_prior_input_size: int = 384,
        dino_prior_grid_size: int = 32,
        dino_prior_num_eigvecs: int = 10,
        dino_prior_chunk_size: int = 8,
        use_dino_graph_bayesian_prior: bool = False,
        dino_bayes_input_size: int = 384,
        dino_bayes_grid_size: int = 32,
        dino_bayes_num_eigvecs: int = 10,
        dino_bayes_temperature: float = 1.0,
        dino_bayes_chunk_size: int = 8,
        **kwargs,
    ):
        super().__init__(args, **kwargs)

        self.w_dino = w_dino
        self.w_dino_merge = w_dino_merge
        self.dino_input_size = dino_input_size
        self.dino_patch_size = dino_patch_size
        # None = apply to all channels; [1] = ch1 only, etc.
        self.dino_channels = dino_channels

        # Build and freeze DINO
        raw_dino = _build_dino(dino_arch, dino_patch_size, dino_checkpoint)
        self.dino = _FrozenModule(raw_dino)

        # Slot for mask logits captured during forward_train
        self._captured_mask_logits: Optional[torch.Tensor] = None

        # DINO graph-partitioning fusion (discussed 260730) -- reuses self.dino
        # (same frozen weights as loss_dino, called at a different resolution).
        self.dino_graph_fusion_head = None
        if use_dino_graph_fusion:
            from models.dino_graph_fusion import DinoGraphFusionHead
            self.dino_graph_fusion_head = DinoGraphFusionHead(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                feat_channels=dino_graph_feat_channels,
                dino_input_size=dino_graph_input_size,
                grid_size=dino_graph_grid_size,
                num_eigvecs=dino_graph_num_eigvecs,
                proj_channels=dino_graph_proj_channels,
                chunk_size=dino_graph_chunk_size,
            )

        # DINO graph E-step fusion (discussed 260730, models/
        # dino_graph_estep_fusion.py) -- separate mechanism, separate
        # attribute, separate config namespace from dino_graph_fusion_head
        # above; deliberately does not share any state with it.
        self.dino_graph_estep_fusion_head = None
        if use_dino_graph_estep_fusion:
            from models.dino_graph_estep_fusion import DinoGraphEStepFusion
            self.dino_graph_estep_fusion_head = DinoGraphEStepFusion(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                dino_input_size=dino_graph_estep_input_size,
                grid_size=dino_graph_estep_grid_size,
                num_eigvecs=dino_graph_estep_num_eigvecs,
                chunk_size=dino_graph_estep_chunk_size,
                temperature=dino_graph_estep_temperature,
                alpha=dino_graph_estep_alpha,
            )

        # DINO graph fusion at feat3 instead of feat0 (discussed 260730) --
        # reuses the exact same DinoGraphFusionHead class as
        # dino_graph_fusion_head above (same zero-init no-op property), just
        # a second instance targeting the deep/semantic scale instead of the
        # shallow/boundary scale. Separate attribute, separate hook.
        self.dino_graph_fusion_deep_head = None
        if use_dino_graph_fusion_deep:
            from models.dino_graph_fusion import DinoGraphFusionHead
            self.dino_graph_fusion_deep_head = DinoGraphFusionHead(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                feat_channels=dino_graph_deep_feat_channels,
                dino_input_size=dino_graph_deep_input_size,
                grid_size=dino_graph_deep_grid_size,
                num_eigvecs=dino_graph_deep_num_eigvecs,
                proj_channels=dino_graph_deep_proj_channels,
                chunk_size=dino_graph_deep_chunk_size,
            )

        # DINO graph BAYESIAN prior (discussed 260811) -- distinct mechanism
        # from dino_graph_fusion_deep_head above: instead of concatenating
        # eigenvectors into a mid-network feature (feat3) and letting later
        # conv/ASPP layers implicitly reprocess them, this computes a genuine
        # categorical prior P(channel | DINO eigenvectors) directly (soft
        # distance to learnable centroids in eigenvector-embedding space,
        # mirroring the paper's own K-Means-on-eigenvectors recipe) and adds
        # its log-probability straight onto decode_head2's raw mask logits
        # at the decision layer -- see models/dino_graph_bayesian_prior.py's
        # docstring for the full design and paper grounding. Separate
        # attribute/hook, not intended to be combined with
        # dino_graph_fusion_deep_head in the same config (both target the
        # same decision, via different mechanisms -- comparing them, not
        # stacking them).
        self.dino_graph_bayesian_prior_head = None
        if use_dino_graph_bayesian_prior:
            from models.dino_graph_bayesian_prior import DinoGraphBayesianPrior
            self.dino_graph_bayesian_prior_head = DinoGraphBayesianPrior(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                num_classes=self.mask_layer,
                dino_input_size=dino_bayes_input_size,
                grid_size=dino_bayes_grid_size,
                num_eigvecs=dino_bayes_num_eigvecs,
                prior_temperature=dino_bayes_temperature,
                chunk_size=dino_bayes_chunk_size,
            )

        # DINO graph attention-gate fusion at feat3 (discussed 260801,
        # models/dino_graph_attention_gate.py) -- alternative fusion TYPE to
        # dino_graph_fusion_deep_head above (concat-based), same feat3
        # injection point. Separate attribute, separate hook -- not intended
        # to be enabled together with use_dino_graph_fusion_deep.
        self.dino_graph_attention_gate_head = None
        if use_dino_graph_attention_gate:
            from models.dino_graph_attention_gate import DinoGraphAttentionGate
            self.dino_graph_attention_gate_head = DinoGraphAttentionGate(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                feat_channels=dino_graph_gate_feat_channels,
                dino_input_size=dino_graph_gate_input_size,
                grid_size=dino_graph_gate_grid_size,
                num_eigvecs=dino_graph_gate_num_eigvecs,
                gate_hidden_channels=dino_graph_gate_hidden_channels,
                chunk_size=dino_graph_gate_chunk_size,
            )

        # DINO graph fusion at feat2, a SECOND independent injection point
        # on top of feat3 (discussed 260801, multi-scale injection) -- reuses
        # DinoGraphFusionHead unchanged, same zero-init no-op property.
        # Separate attribute, separate hook -- meant to be combined with
        # use_dino_graph_fusion_deep (feat3) in the same config.
        self.dino_graph_fusion_deep2_head = None
        if use_dino_graph_fusion_deep2:
            from models.dino_graph_fusion import DinoGraphFusionHead
            self.dino_graph_fusion_deep2_head = DinoGraphFusionHead(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                feat_channels=dino_graph_deep2_feat_channels,
                dino_input_size=dino_graph_deep2_input_size,
                grid_size=dino_graph_deep2_grid_size,
                num_eigvecs=dino_graph_deep2_num_eigvecs,
                proj_channels=dino_graph_deep2_proj_channels,
                chunk_size=dino_graph_deep2_chunk_size,
            )

        # DINO graph eigenvector extraction for decoder-stage injection
        # (discussed 260801, models/dino_graph_eigvec_extractor.py) -- NO
        # trainable params here; the fusion module lives inside decode_head2
        # itself (MultiScaleSegHeadDecoderPrior). Separate attribute,
        # separate hook.
        self.dino_graph_eigvec_extractor = None
        if use_dino_graph_decoder_prior:
            from models.dino_graph_eigvec_extractor import DinoGraphEigvecExtractor
            self.dino_graph_eigvec_extractor = DinoGraphEigvecExtractor(
                dino=self.dino,
                dino_patch_size=dino_patch_size,
                dino_input_size=dino_prior_input_size,
                grid_size=dino_prior_grid_size,
                num_eigvecs=dino_prior_num_eigvecs,
                chunk_size=dino_prior_chunk_size,
            )

    # ------------------------------------------------------------------ #
    # Capture mask logits (same pattern as RCFTissueModel)                #
    # ------------------------------------------------------------------ #
    def _decode_head_forward(self, x, decode_head, flow_feat=None, stem_feat=None, dino_eigvecs=None):
        kwargs = {}
        if flow_feat is not None:
            kwargs['flow_feat'] = flow_feat
        if stem_feat is not None:
            kwargs['stem_feat'] = stem_feat
        if dino_eigvecs is not None:
            kwargs['dino_eigvecs'] = dino_eigvecs
        pred = decode_head.forward(x, **kwargs)
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
    # DINO cross-channel appearance-affinity (soft merge) loss             #
    # ------------------------------------------------------------------ #
    def _dino_merge_loss(self, masks: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
        """
        L_dino_merge: pulls a channel PAIR's DINO centroids together, weighted
        by how much the two channels' soft masks already spatially overlap
        (mask_i * mask_j). This is purely appearance + existing-overlap driven
        — no spatial-position heuristic (e.g. no "ring around instrument =
        tissue" assumption) and no fixed channel identity is required.

        Motivation: motion-based common fate correctly splits an instrument's
        shaft and jaw into separate channels (different motion patterns), but
        eval/appearance treat them as one object. This loss lets channels that
        already border each other (non-trivial soft overlap at the shared
        boundary) AND look visually similar (metal texture, etc.) drift closer
        together in DINO space, without forcing a merge on dissimilar or
        spatially disjoint channel pairs (background vs. tissue, say).

        masks: [B, C, H_m, W_m] soft masks after softmax
        imgs:  [B, 3, H, W]     first-frame training images
        """
        B, C, H_m, W_m = masks.shape
        feats = self._extract_dino_feats(imgs)          # [B, D, H_p, W_p], frozen
        _, D, H_p, W_p = feats.shape

        if H_m != H_p or W_m != W_p:
            masks_p = F.interpolate(masks, (H_p, W_p), mode="bilinear", align_corners=False)
        else:
            masks_p = masks

        centroids = []
        for c in range(C):
            w_c = masks_p[:, c]
            W_c = w_c.sum(dim=(1, 2))
            centroid = (feats * w_c.unsqueeze(1)).sum(dim=(2, 3)) / (W_c.unsqueeze(1) + 1e-6)
            centroids.append(F.normalize(centroid, dim=1))    # [B, D]

        total = torch.tensor(0.0, device=masks.device)
        n_pairs = 0
        for i in range(C):
            for j in range(i + 1, C):
                affinity = (centroids[i] * centroids[j]).sum(dim=1)          # [B], cos sim
                overlap = (masks_p[:, i] * masks_p[:, j]).mean(dim=(1, 2))   # [B], soft co-occurrence
                total = total + (overlap * (1.0 - affinity)).mean()
                n_pairs += 1

        return total / n_pairs

    # ------------------------------------------------------------------ #
    # Override forward_train to append L_dino                             #
    # ------------------------------------------------------------------ #
    def forward_train(self, imgs, seq_ids, seq_names, paths,
                      gt_fw_flows, gt_bw_flows, pl_masks, gaps=None):
        self._captured_mask_logits = None
        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks, gaps=gaps
        )

        if (self.w_dino <= 0.0 and self.w_dino_merge <= 0.0) or self._captured_mask_logits is None:
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

        if self.w_dino > 0.0:
            l_dino = self._dino_consistency_loss(masks_frame0, imgs_frame0)
            losses["loss_dino"] = l_dino
            losses["loss"] = losses["loss"] + self.w_dino * l_dino

        if self.w_dino_merge > 0.0:
            l_dino_merge = self._dino_merge_loss(masks_frame0, imgs_frame0)
            losses["loss_dino_merge"] = l_dino_merge
            losses["loss"] = losses["loss"] + self.w_dino_merge * l_dino_merge

        return losses

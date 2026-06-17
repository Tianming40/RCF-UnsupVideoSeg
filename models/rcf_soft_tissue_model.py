"""
rcf_soft_tissue_model.py — RCFSoftTissueModel

Inherits RCFDinoModel and combines ALL tissue-role losses from
rcf_tissue_model.py with the annotation-driven losses and KL distillation
added for grasp0.  rcf_tissue_model.py is superseded by this file.

Loss set (all weights default to 0.0 — enable in config):

  Inherited from RCFDinoModel:
    L_dino         : DINO visual-consistency (w_dino)

  From RCFTissueModel V2 (unsupervised):
    L_rigid        : instrument channels have uniform internal flow (w_rigid)
    L_grasp_conv   : grasping channel aligns with flow convergence (w_grasp_conv)
    L_deform       : tissue residual > background residual (w_deform)
    L_align        : tissue flow aligns with grasping-channel mean flow (w_align)
    L_motion       : tissue motion > background motion (w_motion)

  Annotation-driven V3 (grasp0-specific):
    L_grasp_flow   : tissue activates where flow matches grasp-point flow,
                     excluding existing instrument region (w_grasp_flow)
    L_dissect_prox : tissue activates near annotated dissection point (w_dissect)

  KL distillation:
    L_distill      : BCE(student_ch, teacher_ch) — preserves pretrained
                     instrument-detector channel from degrading (w_distill)

Usage:
    python main_grasp0.py  configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rcf_dino_model import RCFDinoModel, _FrozenModule
from .rcf_model import RCFModel
from .tissue_role_loss import (
    rigidity_loss,
    grasping_convergence_loss,
    deformation_loss,
    tissue_flow_alignment_loss,
    tissue_motion_loss,
    grasp_flow_alignment_loss,
    dissect_proximity_loss,
    contact_zone_loss,
    tissue_divergence_loss,
    tissue_flow_variance_loss,
    flow_cosine_assignment_loss,
)

import utils
logger = utils.get_logger()


def _gpu_kmeans(pts: torch.Tensor, K: int, n_iter: int = 10) -> torch.Tensor:
    """Euclidean K-means on globally-normalised flow vectors, fully on GPU.

    pts are in [-1,1]^2 (flow / scene_max_magnitude), matching the 2-D space
    that flow_to_color uses to assign colours.  Clusters in this space therefore
    correspond directly to the colour blocks visible in the RAFT visualisation.

    Data-adaptive init (deterministic):
      centroid 0  — mean of near-zero pixels (background, appears white in viz)
      centroids 1..K-1 — mean of pixels inside each of K-1 equal angular sectors,
                         covering the full 360° direction wheel.  This places
                         each initial centroid where the actual data lives in that
                         directional band rather than at a fixed radius of 0.5.

    Args:
        pts    : (M, 2) globally-normalised flow vectors
        K      : number of clusters
        n_iter : EM iterations

    Returns:
        labels (M,) int64, values in [0, K-1]
    """
    M, _  = pts.shape
    device = pts.device
    eps    = 1e-6

    r     = pts.norm(dim=1)                              # [M] per-pixel magnitude
    r_max = r.max().clamp(min=eps)

    # centroid 0: background (near-zero flow → white/grey in flow_to_color)
    bg_mask = r < 0.05 * r_max
    c0 = pts[bg_mask].mean(0) if bg_mask.any() else pts.new_zeros(2)

    # centroids 1..K-1: one per equal angular sector of the direction wheel
    n_dir  = max(K - 1, 1)
    sector = 2.0 * math.pi / n_dir
    theta  = torch.atan2(pts[:, 1], pts[:, 0])          # [M], range [-π, π]
    rest_c = []
    for k in range(n_dir):
        angle_k = -math.pi + (k + 0.5) * sector
        diff    = theta - angle_k
        # wrap angular distance to [-π, π]
        diff    = diff - (2.0 * math.pi) * torch.round(diff / (2.0 * math.pi))
        in_sec  = diff.abs() < sector * 0.5
        if in_sec.any():
            rest_c.append(pts[in_sec].mean(0))
        else:
            # no data in this sector: place at median radius along sector centre
            med_r = r.median()
            rest_c.append(pts.new_tensor([
                med_r.item() * math.cos(angle_k),
                med_r.item() * math.sin(angle_k),
            ]))
    centroids = torch.stack([c0] + rest_c)               # [K, 2]

    # EM iterations
    labels = pts.new_zeros(M, dtype=torch.long)
    for _ in range(n_iter):
        dists     = (pts.unsqueeze(1) - centroids.unsqueeze(0)).norm(dim=2)  # [M, K]
        labels    = dists.argmin(dim=1)                                       # [M]
        one_hot   = pts.new_zeros(M, K).scatter_(1, labels.unsqueeze(1), 1.0)
        counts    = one_hot.sum(0).clamp(min=1)
        centroids = (one_hot.T @ pts) / counts.unsqueeze(1)

    return labels


def flow_cluster_tv_loss(
    masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_channels: list,
    K: int = 3,
    n_iter: int = 10,
    push_margin: float = 0.3,
) -> torch.Tensor:
    """K-means flow clustering guided TV loss on non-instrument channels.

    For each adjacent pixel pair (h or v direction):
      - SAME cluster   → pull:  penalise seg diff          (smooth within region)
      - DIFF cluster   → push:  hinge loss if diff < margin (encourage boundary)

    Both terms act only on non-instrument channels (ch1 excluded).
    Clustering uses global-max-normalised flow (mirrors flow_to_color) so
    background stays near (0,0) and forms its own natural cluster.

    Args:
        masks        : softmax probabilities  (B, C, H, W)
        flow         : optical flow           (B, 2, H, W)
        instrument_channels: channel indices excluded from seg diff
        K            : number of motion clusters
        n_iter       : k-means EM iterations
        push_margin  : hinge margin for cross-cluster boundary encouragement
    """
    B, C, H, W = masks.shape
    eps = 1e-6

    noi_ch    = [c for c in range(C) if c not in instrument_channels]
    masks_noi = masks[:, noi_ch]                          # (B, C', H, W)

    per_sample = []

    for b in range(B):
        flow_b = flow[b]                                  # (2, H, W)

        # global-max normalise: mirrors flow_to_color (dir + relative mag)
        rad_max = flow_b.norm(dim=0).max().clamp(min=eps)
        fn  = flow_b / rad_max                            # (2, H, W)
        pts = fn.permute(1, 2, 0).reshape(-1, 2)         # (H*W, 2)

        with torch.no_grad():
            labels = _gpu_kmeans(pts, K=K, n_iter=n_iter)

        lmap  = labels.reshape(H, W)                      # (H, W)
        same_h = (lmap[:, :-1] == lmap[:, 1:]).float()   # (H, W-1)
        same_v = (lmap[:-1, :] == lmap[1:, :]).float()   # (H-1, W)

        m      = masks_noi[b]                             # (C', H, W)
        diff_h = (m[:, :, :-1] - m[:, :, 1:]).norm(dim=0)   # (H, W-1)
        diff_v = (m[:, :-1, :] - m[:, 1:, :]).norm(dim=0)   # (H-1, W)

        # pull: same cluster → minimise seg diff
        n_same = (same_h.sum() + same_v.sum()).clamp(min=1)
        pull = ((same_h * diff_h).sum() + (same_v * diff_v).sum()) / n_same

        # push: diff cluster → hinge, seg diff should exceed push_margin
        n_diff = ((1 - same_h).sum() + (1 - same_v).sum()).clamp(min=1)
        push = (((1 - same_h) * (push_margin - diff_h).clamp(min=0)).sum() +
                ((1 - same_v) * (push_margin - diff_v).clamp(min=0)).sum()) / n_diff

        per_sample.append(pull + push)

    if not per_sample:
        return masks.sum() * 0
    return torch.stack(per_sample).mean()


def flow_cluster_ce_loss(
    masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_channels: list,
    K: int = 5,
    n_iter: int = 10,
    temperature: float = 0.5,
    instrument_mask: torch.Tensor | None = None,
    diversity_weight: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """K-means color-block CE loss for non-instrument channels.

    Each non-instrument channel softly claims one K-means cluster (= one
    flow_to_color color block).  Assignment is permutation-invariant — no
    fixed channel-to-cluster mapping is assumed.

    K-means labels come purely from RAFT flow (independent of the current
    mask state), so the loss provides non-zero gradient even when all masks
    are uniformly gray — unlike flow_tv (pairwise TV dead zone) and
    flow_cosine (mask-bootstrapped mu dead zone when masks are gray).

    Args:
        masks             : softmax probabilities (B, C, H, W)
        flow              : RAFT flow at mask resolution (B, 2, H, W)
        instrument_channels: excluded from both pixel weights and channel pool
        K                 : K-means clusters — should match flow_tv_K so both
                            losses operate on the same color-block partition
        n_iter            : K-means EM iterations
        temperature       : softmax temperature for cluster-channel assignment
                            (lower = harder claiming; 0.1–0.5 recommended)
        instrument_mask   : [B, H, W] teacher instrument prob — preferred over
                            student mask to suppress instrument pixels (avoids
                            feedback loop if student ch1 drifts)
        diversity_weight  : pushes channel affinity profiles apart in cluster
                            space (same role as flow_cosine_diversity)
        eps               : numerical stability
    """
    B, C, H, W = masks.shape
    non_inst = [c for c in range(C) if c not in instrument_channels]
    K_ch     = len(non_inst)

    # ── instrument pixel suppression ────────────────────────────────────────
    if instrument_mask is not None:
        inst_w = instrument_mask.detach().clamp(0, 1)
    else:
        inst_w = sum(masks[:, c] for c in instrument_channels).clamp(0, 1).detach()
    non_inst_w = 1.0 - inst_w                               # [B, H, W]

    # ── pixel weights: non-instrument AND moving (static pixels unhelpful) ──
    flow_mag = flow.norm(dim=1)                              # [B, H, W]
    pixel_w  = non_inst_w * flow_mag                        # [B, H, W]

    # ── non-instrument student masks, re-normalised to sum=1 among them ─────
    student = torch.stack([masks[:, c] for c in non_inst], dim=1)  # [B, K_ch, H, W]
    student = student / (student.sum(dim=1, keepdim=True) + eps)

    total_ce  = torch.tensor(0.0, device=masks.device)
    total_div = torch.tensor(0.0, device=masks.device)
    n_pairs   = K_ch * (K_ch - 1) // 2

    for b in range(B):
        flow_b  = flow[b]                                    # [2, H, W]
        rad_max = flow_b.norm(dim=0).max().clamp(min=eps)
        pts     = (flow_b / rad_max).permute(1, 2, 0).reshape(H * W, 2)

        with torch.no_grad():
            labels = _gpu_kmeans(pts, K=K, n_iter=n_iter)   # [H*W]
        lmap = labels.reshape(H, W)                          # [H, W]

        pw      = pixel_w[b]                                 # [H, W]
        w_total = pw.sum() + eps

        # ── affinity matrix P[K_ch, K] ───────────────────────────────────────
        # For each (channel, cluster): weighted mean of channel mask in cluster.
        # one-hot cluster labels → [K, H*W]
        oh_flat = F.one_hot(labels, K).float().T             # [K, H*W]
        # cluster-weighted pixel weights → [K, H*W]
        cw_flat = oh_flat * pw.reshape(1, H * W)             # [K, H*W]
        denom   = cw_flat.sum(dim=1).clamp(min=eps)          # [K]
        # P[K_ch, K] = student_flat @ cw_flat.T / denom  (detached for target)
        s_flat  = student[b].detach().reshape(K_ch, H * W)
        P       = (s_flat @ cw_flat.T) / denom.unsqueeze(0)  # [K_ch, K]

        # ── soft assignment: channels compete per cluster (softmax over K_ch) ─
        A = F.softmax(P / temperature, dim=0).detach()       # [K_ch, K]

        # ── pixel-level CE targets from cluster assignment ───────────────────
        target   = A[:, lmap]                                 # [K_ch, H, W]
        log_s    = torch.log(student[b] + eps)                # [K_ch, H, W]
        ce       = -(target * log_s).sum(dim=0)               # [H, W]
        total_ce = total_ce + (ce * pw / w_total).sum()

        # ── diversity: push channel mean flow directions apart ───────────────
        # Uses mu-based approach (same as flow_cosine_diversity): gradient
        # = d(mu_c)/d(mask_c[pixel]) ∝ (flow_dir[pixel] - mu_c), which is
        # non-zero from per-pixel flow variation even when masks are gray.
        if diversity_weight > 0 and n_pairs > 0:
            flow_dir = flow_b / (flow_mag[b].unsqueeze(0) + eps)  # [2, H, W]
            mu_list_g = []
            for ci in range(K_ch):
                m = student[b, ci] * pixel_w[b]               # [H, W], no detach
                w = m.sum() + eps
                mu_x = (flow_dir[0] * m).sum() / w
                mu_y = (flow_dir[1] * m).sum() / w
                mu = torch.stack([mu_x, mu_y])                 # [2]
                mu_list_g.append(mu / (mu.norm() + eps))
            div_b = sum(
                (mu_list_g[i] * mu_list_g[j]).sum()
                for i in range(K_ch) for j in range(i + 1, K_ch)
            ) / n_pairs
            total_div = total_div + div_b

    total_ce  = total_ce  / B
    total_div = total_div / B
    return total_ce + diversity_weight * total_div


_DINO_ONLY_KWARGS = frozenset({
    'dino_arch', 'dino_patch_size', 'dino_checkpoint', 'dino_input_size', 'w_dino',
})


class RCFSoftTissueModel(RCFDinoModel):
    """
    RCFDinoModel + full tissue-role loss set + KL distillation.

    All tissue/annotation kwargs are keyword-only with safe defaults.
    Pass them under  model_kwargs:  in the YAML config.

    Channel convention (grasp0 fine-tune):
      instrument_channel  = 1   (pretrained grasp10 instrument, protected by L_distill)
      instrument_channels = [1] (for L_rigid)
      tissue_channel      = 2   (new soft-tissue channel to train)
      grasping_channel    = 1   (same as instrument for L_align / L_grasp_conv)
      bg_channels         = [0, 3, 4]
      distill_channel     = 1   (same as instrument_channel)
    """

    def __init__(
        self,
        args,
        # ── channel assignments ──────────────────────────────────────────
        instrument_channels=(1,),
        instrument_channel: int = 1,
        tissue_channel: int = 2,
        grasping_channel: int = 1,
        bg_channels=(0, 3, 4),
        # ── V2 unsupervised losses ────────────────────────────────────────
        w_rigid: float = 0.0,
        w_grasp_conv: float = 0.0,
        w_deform: float = 0.0,
        w_align: float = 0.0,
        w_motion: float = 0.0,
        motion_margin: float = 1.0,
        deform_margin: float = 0.5,
        min_grasp_frac: float = 0.005,
        # ── V3 annotation-driven losses ───────────────────────────────────
        w_grasp_flow: float = 0.0,
        w_dissect: float = 0.0,
        grasp_flow_min_mag: float = 0.5,
        dissect_sigma: float = 0.12,
        # ── KL distillation ───────────────────────────────────────────────
        w_distill: float = 0.0,
        teacher_ckpt: Optional[str] = None,
        distill_channel: int = 1,
        distill_mode: str = 'one_sided',  # 'one_sided' (relu) | 'symmetric' (MSE)
        # distill annealing (v13): after distill_warmup_epochs, switch mode/weight
        distill_warmup_epochs: int = 0,       # 0 = no annealing
        distill_cool_mode: str = 'one_sided', # mode after warmup
        w_distill_cool: float = 0.5,          # weight after warmup
        # ── V4 spatial / divergence losses (default off) ──────────────────
        w_contact: float = 0.0,
        contact_dilation_r: int = 10,
        w_div_tissue: float = 0.0,
        # ── V5 flow-variance tissue loss (default off) ────────────────────
        w_rigid_tissue: float = 0.0,
        # ── V8 flow-cosine assignment loss (default off) ──────────────────
        w_flow_cosine: float = 0.0,
        flow_cosine_temperature: float = 0.5,
        flow_cosine_diversity: float = 0.0,
        # ── V10/V11 head reset (default off) ─────────────────────────────
        reset_non_instrument_heads: bool = False,  # v10: reset only conv_seg rows
        reset_full_decode_head: bool = False,       # v11: reset entire decode_head2
        # ── V14 flow-cluster TV loss (default off) ────────────────────────
        w_flow_tv: float = 0.0,
        flow_tv_K: int = 3,              # number of motion clusters
        flow_tv_n_iter: int = 10,        # k-means iterations
        flow_tv_start_epoch: int = 0,    # delay TV loss until this epoch
        flow_tv_push_margin: float = 0.3,  # hinge margin for cross-cluster push
        # ── V16 flow-cluster CE loss (default off) ────────────────────────
        w_flow_cluster_ce: float = 0.0,
        flow_cluster_ce_temperature: float = 0.3,   # lower = harder channel claiming
        flow_cluster_ce_diversity: float = 0.5,     # push channels' cluster profiles apart
        flow_cluster_ce_start_epoch: int = 0,
        # ── backward compat with old configs ─────────────────────────────
        w_compact: float = 0.0,
        **kwargs,
    ):
        import copy
        # save before super().__init__ mutates backbone2/decode_head dicts
        # (create_backbone_with_ema does dict.pop('type') in-place)
        _kwargs_for_teacher = copy.deepcopy(kwargs)
        super().__init__(args, **kwargs)

        self.instrument_channels = list(instrument_channels)
        self.instrument_channel  = instrument_channel
        self.tissue_channel      = tissue_channel
        self.grasping_channel    = grasping_channel
        self.bg_channels         = tuple(bg_channels)

        self.w_rigid         = w_rigid
        self.w_grasp_conv    = w_grasp_conv
        self.w_deform        = w_deform
        self.w_align         = w_align
        self.w_motion        = w_motion
        self.motion_margin   = motion_margin
        self.deform_margin   = deform_margin
        self.min_grasp_frac  = min_grasp_frac

        self.w_grasp_flow       = w_grasp_flow
        self.w_dissect          = w_dissect
        self.grasp_flow_min_mag = grasp_flow_min_mag
        self.dissect_sigma      = dissect_sigma

        self.w_distill             = w_distill
        self.distill_channel       = distill_channel
        self.distill_mode          = distill_mode
        self.distill_warmup_epochs = distill_warmup_epochs
        self.distill_cool_mode     = distill_cool_mode
        self.w_distill_cool        = w_distill_cool
        self._distill_cooled       = False

        self.w_contact         = w_contact
        self.contact_dilation_r = contact_dilation_r
        self.w_div_tissue      = w_div_tissue
        self.w_rigid_tissue          = w_rigid_tissue
        self.w_flow_cosine              = w_flow_cosine
        self.flow_cosine_temperature    = flow_cosine_temperature
        self.flow_cosine_diversity      = flow_cosine_diversity
        self.reset_non_instrument_heads = reset_non_instrument_heads
        self.reset_full_decode_head     = reset_full_decode_head
        self.w_flow_tv           = w_flow_tv
        self.flow_tv_K           = flow_tv_K
        self.flow_tv_n_iter      = flow_tv_n_iter
        self.flow_tv_start_epoch = flow_tv_start_epoch
        self.flow_tv_push_margin = flow_tv_push_margin
        self._flow_tv_active     = (flow_tv_start_epoch == 0)  # active immediately if no delay

        self.w_flow_cluster_ce           = w_flow_cluster_ce
        self.flow_cluster_ce_temperature = flow_cluster_ce_temperature
        self.flow_cluster_ce_diversity   = flow_cluster_ce_diversity
        self.flow_cluster_ce_start_epoch = flow_cluster_ce_start_epoch
        self._flow_cluster_ce_active     = (flow_cluster_ce_start_epoch == 0)

        # per-batch state (set in forward, consumed in forward_train)
        self._batch_grasp_xy      = None
        self._batch_dissect_xy    = None
        self._captured_residual_fw = None

        # frozen teacher for KL distillation
        self._teacher = None
        if w_distill > 0 and teacher_ckpt is not None:
            self._teacher = self._build_frozen_teacher(teacher_ckpt, _kwargs_for_teacher)
            logger.info("[RCFSoftTissueModel] Distill teacher: %s", teacher_ckpt)

        logger.info(
            "[RCFSoftTissueModel] "
            "inst=%s  inst_ch=%d  tissue=%d  grasp=%d  bg=%s\n"
            "  w_rigid=%.3f  w_grasp_conv=%.3f  w_deform=%.3f  "
            "w_align=%.3f  w_motion=%.3f\n"
            "  w_grasp_flow=%.3f  w_dissect=%.3f  w_distill=%.3f",
            self.instrument_channels, instrument_channel,
            tissue_channel, grasping_channel, list(self.bg_channels),
            w_rigid, w_grasp_conv, w_deform, w_align, w_motion,
            w_grasp_flow, w_dissect, w_distill,
        )

    # ── Head reset (V10) ──────────────────────────────────────────────────────

    def _reset_non_instrument_mask_heads(self) -> None:
        """
        Re-initialise the cls_seg conv rows for every non-instrument channel.

        Only the final 1×1 classification conv of decode_head2 is touched;
        the backbone and feature-extraction layers keep their pretrained weights.
        Channel `instrument_channels` rows are left intact so the distill loss
        can immediately enforce instrument segmentation from epoch 0.
        """
        conv_seg = self.decode_head2.conv_seg   # Conv2d(feat_ch, num_classes, 1)
        non_inst = [c for c in range(conv_seg.weight.shape[0])
                    if c not in self.instrument_channels]
        with torch.no_grad():
            for c in non_inst:
                nn.init.kaiming_uniform_(conv_seg.weight[c: c + 1])
                if conv_seg.bias is not None:
                    nn.init.zeros_(conv_seg.bias[c: c + 1])
        logger.info(
            "[RCFSoftTissueModel] reset_non_instrument_heads: "
            "re-initialised conv_seg rows %s  (kept rows %s)",
            non_inst, self.instrument_channels,
        )

    def _reset_full_decode_head2(self) -> None:
        """
        Re-initialise ALL parameters of decode_head2 (convs.0, convs.1, conv_seg).

        Resets the entire segmentation pathway, not just the final 1×1 conv.
        This removes the grasp10 tissue-partition prior embedded in the intermediate
        feature-extraction layers (convs.0/1), which are shared across all channels
        and cannot be selectively reset per channel.

        ch1 recovery is guaranteed by the BCE distill loss (symmetric mode), which
        has gradient ∝ 1/student_p → very large near zero → always dominates.
        """
        for m in self.decode_head2.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        logger.info(
            "[RCFSoftTissueModel] reset_full_decode_head: "
            "reset all parameters of decode_head2 (convs.0, convs.1, conv_seg)"
        )

    # ── Distill annealing (V13) ───────────────────────────────────────────────

    def set_distill_cool(self) -> None:
        """
        Switch distill from warmup mode (symmetric, high weight) to cool mode
        (one_sided, low weight).  Called by TissueModel.on_train_epoch_start
        once self.current_epoch >= distill_warmup_epochs.
        No-op if already switched or if distill_warmup_epochs == 0.
        """
        if self._distill_cooled:
            return
        logger.info(
            "[RCFSoftTissueModel] distill annealing: %s w=%.2f → %s w=%.2f",
            self.distill_mode, self.w_distill,
            self.distill_cool_mode, self.w_distill_cool,
        )
        self.distill_mode  = self.distill_cool_mode
        self.w_distill     = self.w_distill_cool
        self._distill_cooled = True

    # ── Frozen teacher ─────────────────────────────────────────────────────────

    def _build_frozen_teacher(self, ckpt_path: str, base_kwargs: dict) -> nn.Module:
        """
        Lightweight teacher: RCFModel (backbone2 + decode_head2, no DINO).
        Loaded from a PL checkpoint with strict=False.
        Wrapped in _FrozenModule so it stays in eval mode permanently.
        """
        rcf_kwargs = {k: v for k, v in base_kwargs.items()
                      if k not in _DINO_ONLY_KWARGS}
        teacher = RCFModel(args=self.args, **rcf_kwargs)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd   = ckpt.get('state_dict', ckpt)
        if any(k.startswith('model.') for k in sd):
            sd = {k[len('model.'):]: v for k, v in sd.items()
                  if k.startswith('model.')}
        m = teacher.load_state_dict(sd, strict=False)
        logger.info("[Teacher] missing=%d  unexpected=%d",
                    len(m.missing_keys), len(m.unexpected_keys))
        return _FrozenModule(teacher)

    # ── Capture annotation tensors ─────────────────────────────────────────────

    def forward(self, x, return_pred_vis_list=False):
        if self.training:
            self._batch_grasp_xy   = x.get('grasp_xy',   None)
            self._batch_dissect_xy = x.get('dissect_xy', None)
        return super().forward(x, return_pred_vis_list=return_pred_vis_list)

    # ── Capture residual (needed for L_deform) ─────────────────────────────────

    def pred_separate_residual(self, feats, batch_size, im_num):
        res_fw, res_bw = super().pred_separate_residual(feats, batch_size, im_num)
        if self.training:
            self._captured_residual_fw = res_fw   # [B, 2*C, fH, fW]
        return res_fw, res_bw

    # ── Teacher ch-prob ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _teacher_ch_prob(self, imgs_b3hw: torch.Tensor) -> torch.Tensor:
        """backbone2 + decode_head2 of frozen teacher → softmax [B, C, Hm, Wm]."""
        tm    = self._teacher.module
        feats = tm.extract_feat(imgs_b3hw, tm.backbone2)
        logits = tm._decode_head_forward(feats, tm.decode_head2)
        if logits.shape[-2:] != torch.Size(list(self.mask_size)):
            logits = self.resize(logits, self.mask_size)
        return F.softmax(logits, dim=1)

    # ── forward_train ──────────────────────────────────────────────────────────

    def forward_train(self, imgs, seq_ids, seq_names, paths,
                      gt_fw_flows, gt_bw_flows, pl_masks):
        # super() = RCFDinoModel which:
        #   1. resets _captured_mask_logits
        #   2. runs RCFModel.forward_train (sets _captured_mask_logits, calls
        #      pred_separate_residual which sets _captured_residual_fw)
        #   3. appends L_dino
        #   4. returns WITHOUT clearing _captured_mask_logits
        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks,
        )

        if self._captured_mask_logits is None:
            return losses

        # ── reconstruct soft masks ─────────────────────────────────────────────
        B, I = imgs.shape[0], imgs.shape[1]
        raw = self._captured_mask_logits        # [B*I, C, fH, fW]
        self._captured_mask_logits = None

        C = raw.shape[1]
        if self.allow_mask_resize and raw.shape[-2:] != torch.Size(list(self.mask_size)):
            raw = self.resize(raw, self.mask_size)
        raw    = raw.view(B, I, C, *raw.shape[-2:])
        masks  = F.softmax(raw, dim=2)          # [B, I, C, Hm, Wm]
        masks0 = masks[:, 0]                    # [B, C, Hm, Wm]  frame-0

        # ── forward flow at mask resolution ───────────────────────────────────
        flow = gt_fw_flows[:, 0]
        if flow.ndim == 4 and flow.shape[-1] == 2:
            flow = flow.permute(0, 3, 1, 2).contiguous()
        flow_r   = self.resize(flow, self.mask_size)   # [B, 2, Hm, Wm]
        flow_mag = flow_r.norm(dim=1)                  # [B, Hm, Wm]

        # ── annotation tensors ────────────────────────────────────────────────
        grasp_xy   = self._batch_grasp_xy
        dissect_xy = self._batch_dissect_xy
        self._batch_grasp_xy   = None
        self._batch_dissect_xy = None

        # ── 1. L_rigid ────────────────────────────────────────────────────────
        if self.w_rigid > 0:
            L = rigidity_loss(masks0, flow_r, self.instrument_channels)
            losses['loss_rigid'] = L
            losses['loss'] = losses['loss'] + self.w_rigid * L

        # ── 2. L_grasp_conv ───────────────────────────────────────────────────
        if self.w_grasp_conv > 0:
            L = grasping_convergence_loss(masks0, flow_r, self.grasping_channel)
            losses['loss_grasp_conv'] = L
            losses['loss'] = losses['loss'] + self.w_grasp_conv * L

        # ── 3. L_deform ───────────────────────────────────────────────────────
        res_fw = self._captured_residual_fw
        self._captured_residual_fw = None
        if self.w_deform > 0 and res_fw is not None:
            if res_fw.shape[-2:] != torch.Size(list(self.mask_size)):
                res_fw = self.resize(res_fw, self.mask_size)
            L = deformation_loss(
                masks0, res_fw,
                instrument_channels=self.instrument_channels,
                n_classes=self.num_classes,
                pred_div_coeff=self.decode_head.pred_div_coeff,
                residual_scale=self.decode_head.residual_adjustment_scale,
            )
            losses['loss_deform'] = L
            losses['loss'] = losses['loss'] + self.w_deform * L

        # ── 4. L_align ────────────────────────────────────────────────────────
        if self.w_align > 0:
            L = tissue_flow_alignment_loss(
                masks0, flow_r,
                grasping_channel=self.grasping_channel,
                tissue_channel=self.tissue_channel,
                min_grasp_frac=self.min_grasp_frac,
            )
            losses['loss_align'] = L
            losses['loss'] = losses['loss'] + self.w_align * L

        # ── 5. L_motion ───────────────────────────────────────────────────────
        if self.w_motion > 0:
            L = tissue_motion_loss(
                masks0, flow_mag,
                tissue_channel=self.tissue_channel,
                bg_channels=self.bg_channels,
                margin=self.motion_margin,
            )
            losses['loss_motion'] = L
            losses['loss'] = losses['loss'] + self.w_motion * L

        # ── 6. L_grasp_flow ───────────────────────────────────────────────────
        if self.w_grasp_flow > 0 and grasp_xy is not None:
            L = grasp_flow_alignment_loss(
                masks0, flow_r,
                tissue_channel=self.tissue_channel,
                instrument_channel=self.instrument_channel,
                grasp_xy=grasp_xy,
                min_flow_mag=self.grasp_flow_min_mag,
            )
            losses['loss_grasp_flow'] = L
            losses['loss'] = losses['loss'] + self.w_grasp_flow * L

        # ── 7. L_dissect_prox ─────────────────────────────────────────────────
        if self.w_dissect > 0 and dissect_xy is not None:
            L = dissect_proximity_loss(
                masks0,
                tissue_channel=self.tissue_channel,
                dissect_xy=dissect_xy,
                sigma=self.dissect_sigma,
            )
            losses['loss_dissect'] = L
            losses['loss'] = losses['loss'] + self.w_dissect * L

        # ── teacher forward (shared by L_distill block-8 and L_flow_cosine block-12)
        teacher_p = None
        if self._teacher is not None and (self.w_distill > 0 or self.w_flow_cosine > 0):
            with torch.no_grad():
                teacher_p = self._teacher_ch_prob(imgs[:, 0])   # [B, C, Hm, Wm]

        # ── 8. L_distill ──────────────────────────────────────────────────────
        if self.w_distill > 0 and teacher_p is not None:
            tc        = self.distill_channel
            student_p = masks0[:, tc].clamp(1e-6, 1 - 1e-6)
            target_p  = teacher_p[:, tc].detach()
            if self.distill_mode == 'symmetric':
                # BCE: gradient ∝ (teacher/student), explodes near 0 → dominates
                # other losses even when ch1 starts random.  This is what v12/v13
                # used implicitly (old code always had BCE).  MSE was weaker and
                # caused ch1 to not recover against competing warp_seg gradients.
                L = F.binary_cross_entropy(student_p, target_p)
            else:
                # one-sided relu: only penalise student < teacher.
                # Use when ch1 is mostly intact and we only want to prevent dropout.
                L = F.relu(target_p - student_p).mean()
            losses['loss_distill'] = L
            losses['loss'] = losses['loss'] + self.w_distill * L

        # ── 9. L_contact ──────────────────────────────────────────────────────
        if self.w_contact > 0:
            # channel-agnostic: all non-instrument channels are candidates
            non_inst = [c for c in range(C) if c not in self.instrument_channels]
            L = contact_zone_loss(
                masks0,
                instrument_channel=self.instrument_channel,
                tissue_channels=non_inst,
                dilation_r=self.contact_dilation_r,
            )
            losses['loss_contact'] = L
            losses['loss'] = losses['loss'] + self.w_contact * L

        # ── 10. L_div_tissue ──────────────────────────────────────────────────
        if self.w_div_tissue > 0:
            non_inst = [c for c in range(C) if c not in self.instrument_channels]
            L = tissue_divergence_loss(
                masks0,
                flow=flow_r,
                tissue_channels=non_inst,
                instrument_channels=list(self.instrument_channels),
            )
            losses['loss_div_tissue'] = L
            losses['loss'] = losses['loss'] + self.w_div_tissue * L

        # ── 11. L_rigid_tissue ────────────────────────────────────────────────
        if self.w_rigid_tissue > 0:
            non_inst = [c for c in range(C) if c not in self.instrument_channels]
            L = tissue_flow_variance_loss(masks0, flow=flow_r, tissue_channels=non_inst)
            losses['loss_rigid_tissue'] = L
            losses['loss'] = losses['loss'] + self.w_rigid_tissue * L

        # ── 12. L_flow_cosine ─────────────────────────────────────────────────
        if self.w_flow_cosine > 0:
            # use teacher's instrument channel as exclusion mask: breaks the
            # feedback loop where student ch1 drift → instrument pixels no longer
            # excluded → flow_cosine pulls them into tissue channels
            t_inst_mask = (teacher_p[:, self.distill_channel]
                           if teacher_p is not None else None)
            L = flow_cosine_assignment_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                temperature=self.flow_cosine_temperature,
                instrument_mask=t_inst_mask,
                diversity_weight=self.flow_cosine_diversity,
            )
            losses['loss_flow_cosine'] = L
            losses['loss'] = losses['loss'] + self.w_flow_cosine * L

        # ── 13. L_flow_tv ─────────────────────────────────────────────────────
        if self.w_flow_tv > 0 and self._flow_tv_active:
            L = flow_cluster_tv_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                K=self.flow_tv_K,
                n_iter=self.flow_tv_n_iter,
                push_margin=self.flow_tv_push_margin,
            )
            losses['loss_flow_tv'] = L
            losses['loss'] = losses['loss'] + self.w_flow_tv * L

        # ── 14. L_flow_cluster_ce ─────────────────────────────────────────────
        if self.w_flow_cluster_ce > 0 and self._flow_cluster_ce_active:
            t_inst_mask = (teacher_p[:, self.distill_channel]
                           if teacher_p is not None else None)
            L = flow_cluster_ce_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                K=self.flow_tv_K,
                n_iter=self.flow_tv_n_iter,
                temperature=self.flow_cluster_ce_temperature,
                instrument_mask=t_inst_mask,
                diversity_weight=self.flow_cluster_ce_diversity,
            )
            losses['loss_flow_cluster_ce'] = L
            losses['loss'] = losses['loss'] + self.w_flow_cluster_ce * L

        return losses

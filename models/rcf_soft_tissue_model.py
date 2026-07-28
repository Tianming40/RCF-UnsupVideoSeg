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
import torchvision
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

_cluster_ce_call_count = 0  # for periodic debug logging


def _gpu_kmeans(pts: torch.Tensor, K: int, max_iter: int = 300) -> torch.Tensor:
    """K-means++ init + EM until convergence on GPU.

    pts can be any D-dimensional feature vector (e.g. 2-D normalised flow,
    or 5-D joint flow+HSV).  K-means++ seeding gives better initialisation
    than a fixed ring and typically converges in <50 iterations.

    Args:
        pts      : (M, D) feature vectors
        K        : number of clusters
        max_iter : maximum EM steps (stops early if labels stop changing)

    Returns:
        labels (M,) int64, values in [0, K-1]
    """
    M, _   = pts.shape
    device = pts.device

    # ── k-means++ init ────────────────────────────────────────────────────────
    first = torch.randint(M, (1,), device=device).item()
    centroids = [pts[first]]
    for _ in range(K - 1):
        c   = torch.stack(centroids)                               # [k, D]
        d2  = (pts.unsqueeze(1) - c.unsqueeze(0)).norm(dim=2).min(dim=1).values ** 2
        idx = torch.multinomial(d2 / d2.sum(), 1).item()
        centroids.append(pts[idx])
    centroids = torch.stack(centroids)                             # [K, D]

    # ── EM until convergence ──────────────────────────────────────────────────
    labels = pts.new_zeros(M, dtype=torch.long)
    for _ in range(max_iter):
        dists      = (pts.unsqueeze(1) - centroids.unsqueeze(0)).norm(dim=2)  # [M, K]
        new_labels = dists.argmin(dim=1)                                       # [M]
        if (new_labels == labels).all():
            break
        labels    = new_labels
        one_hot   = pts.new_zeros(M, K).scatter_(1, labels.unsqueeze(1), 1.0)
        counts    = one_hot.sum(0).clamp(min=1)
        centroids = (one_hot.T @ pts) / counts.unsqueeze(1)

    return labels


def _rgb_to_hsv_features(img_bchw: torch.Tensor) -> torch.Tensor:
    """RGB image → circular-hue + saturation features.

    Args:
        img_bchw : [B, 3, H, W] float in [0, 1] or [0, 255]

    Returns:
        [B, H*W, 3] — (cos(2π·h), sin(2π·h), s), all in [-1, 1]
    """
    img = img_bchw.float()
    if img.max() > 2.0:
        img = img / 255.0
    eps = 1e-6
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    max_c = img.max(dim=1).values
    delta = (max_c - img.min(dim=1).values).clamp(min=eps)

    h = torch.zeros_like(r)
    mask_r = (max_c == r)
    mask_g = (max_c == g) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] =  (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] =  (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0 * 2.0 * math.pi                                   # [0, 2π]
    s = delta / max_c.clamp(min=eps) * (max_c > eps).float()

    B, H, W = h.shape
    feats = torch.stack([torch.cos(h), torch.sin(h), s], dim=1)   # [B, 3, H, W]
    return feats.reshape(B, 3, H * W).permute(0, 2, 1)            # [B, N, 3]


def _sinkhorn(log_alpha: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """Sinkhorn row/column normalisation in log space → doubly-stochastic matrix.

    Args:
        log_alpha : [R, C] unnormalised log assignment matrix (e.g. P / temperature)
        n_iter    : alternating normalisation steps (20 is enough for sharp results)

    Returns:
        [R, C] doubly-stochastic matrix (row sums ≈ 1, col sums ≈ 1)
    """
    for _ in range(n_iter):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=0, keepdim=True)
    return log_alpha.exp()


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
    max_iter: int = 300,
    temperature: float = 0.5,
    instrument_mask: torch.Tensor | None = None,
    diversity_weight: float = 0.0,
    image: torch.Tensor | None = None,
    color_weight: float = 1.0,
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
    global _cluster_ce_call_count
    _cluster_ce_call_count += 1
    do_debug = (_cluster_ce_call_count % 150 == 1)  # print once every ~150 batches

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
    lmap_vis  = None  # cluster label map for b=0, returned for visualization

    for b in range(B):
        flow_b     = flow[b]                                 # [2, H, W]
        flow_mag_px = flow_b.norm(dim=0, keepdim=True).clamp(min=eps)
        # Unit-direction vectors: cluster boundaries align with RAFT hue blocks
        pts = (flow_b / flow_mag_px).permute(1, 2, 0).reshape(H * W, 2)

        with torch.no_grad():
            # Color features removed: they introduce image-texture bias and
            # break alignment with the RAFT flow visualization.
            pts_joint = pts
            labels = _gpu_kmeans(pts_joint, K=K, max_iter=max_iter)  # [H*W]
            # Re-label clusters by ascending flow magnitude so that cluster 0
            # always = most-static (background) and cluster K-1 = most-moving
            # (instrument).  Without this, k-means++ random init permutes the
            # cluster IDs per image → Sinkhorn assigns the same channel to
            # opposite motion types across batches → gradients cancel → gray mask.
            oh_s   = pts.new_zeros(H * W, K).scatter_(1, labels.unsqueeze(1), 1.0)
            flow_c = (oh_s.T @ pts) / oh_s.sum(0).clamp(min=1).unsqueeze(1)  # [K, 2]
            norms  = flow_c.norm(dim=1)                                        # [K]
            order  = norms.argsort()                                           # ascending
            remap  = torch.zeros(K, dtype=torch.long, device=pts.device)
            remap[order] = torch.arange(K, device=pts.device)
            labels = remap[labels]
            if do_debug and b == 0:
                sizes = oh_s.sum(0)[order].long()                              # sorted sizes
                logger.info(
                    f"[cluster_ce #{_cluster_ce_call_count}] "
                    f"flow norms (sorted): {norms[order].tolist()} "
                    f"cluster sizes: {sizes.tolist()} "
                    f"assign: cluster0(static)->dummy, "
                    + ", ".join(f"cluster{k+1}->ch{non_inst[k]}" for k in range(K_ch))
                )
        lmap = labels.reshape(H, W)                          # [H, W]
        if b == 0:
            lmap_vis = lmap.detach().cpu()

        pw      = pixel_w[b]                                 # [H, W]
        w_total = pw.sum() + eps

        # ── affinity matrix P[K_ch, K] ───────────────────────────────────────
        # P[ci, k] = weighted-mean of channel ci's mask over cluster-k pixels.
        oh_flat = F.one_hot(labels, K).float().T             # [K, H*W]
        cw_flat = oh_flat * pw.reshape(1, H * W)             # [K, H*W]
        denom   = cw_flat.sum(dim=1).clamp(min=eps)          # [K]
        s_flat  = student[b].detach().reshape(K_ch, H * W)
        P       = (s_flat @ cw_flat.T) / denom.unsqueeze(0)  # [K_ch, K]

        # ── Fixed direct assignment (canonical sort → direct map) ────────────
        # After sorting clusters by flow magnitude (cluster 0 = most static,
        # cluster K-1 = most moving), assign each non-instrument channel to
        # the cluster with the matching rank, skipping cluster 0 (static):
        #
        #   cluster 0 (static) → dummy   (pixel_w ≈ 0 anyway; no signal lost)
        #   cluster 1          → non_inst[0]
        #   cluster 2          → non_inst[1]
        #   ...
        #   cluster K_ch       → non_inst[K_ch-1]
        #
        # This is completely independent of the current mask state (no P
        # needed), so the CE target is stable from the very first batch and
        # gradients accumulate consistently instead of cancelling.
        with torch.no_grad():
            A_noi = P.new_zeros(K_ch, K)
            for k in range(K_ch):
                A_noi[k, k + 1] = 1.0                            # [K_ch, K]

        # ── pixel-level CE with one-hot targets ──────────────────────────────
        target   = A_noi[:, lmap]                                # [K_ch, H, W]
        log_s    = torch.log(student[b] + eps)                   # [K_ch, H, W]
        ce       = -(target * log_s).sum(dim=0)                  # [H, W]
        batch_ce = (ce * pw / w_total).sum()
        total_ce = total_ce + batch_ce
        if do_debug and b == 0:
            with torch.no_grad():
                # per-channel: what is the student probability at assigned pixels?
                ch_probs = []
                for k in range(K_ch):
                    mask_k = (lmap == k + 1)                     # pixels for cluster k+1
                    if mask_k.sum() > 0:
                        ch_probs.append(f"ch{non_inst[k]}@cl{k+1}={student[b,k][mask_k].mean():.3f}")
                logger.info(
                    f"[cluster_ce #{_cluster_ce_call_count}] "
                    f"ce={batch_ce.item():.4f}  "
                    f"student probs: {' '.join(ch_probs)}"
                )

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
    return total_ce + diversity_weight * total_div, lmap_vis


_flow_angle_ce_call_count = 0

def flow_angle_ce_loss(
    masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_channels: list,
    instrument_mask: torch.Tensor | None = None,
    diversity_weight: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Direct flow-angle CE loss — replaces k-means with atan2 sector partition.

    Divides [-pi, pi] into K_ch equal sectors (K_ch = number of non-instrument
    channels).  Each sector maps to one non-instrument channel.  This directly
    mirrors the RAFT flow visualisation hue, so sector boundaries = color-block
    boundaries in the gt_flow image.

    No k-means, no dummy cluster, fully deterministic and consistent across batches.
    Static pixels are suppressed via pixel_w = non_inst_w * flow_mag.
    """
    global _flow_angle_ce_call_count
    _flow_angle_ce_call_count += 1
    do_debug = (_flow_angle_ce_call_count % 150 == 1)

    B, C, H, W = masks.shape
    non_inst = [c for c in range(C) if c not in instrument_channels]
    K_ch = len(non_inst)

    # instrument + static pixel suppression
    if instrument_mask is not None:
        inst_w = instrument_mask.detach().clamp(0, 1)
    else:
        inst_w = sum(masks[:, c] for c in instrument_channels).clamp(0, 1).detach()
    non_inst_w = 1.0 - inst_w                               # [B, H, W]
    flow_mag   = flow.norm(dim=1)                            # [B, H, W]
    pixel_w    = non_inst_w * flow_mag                       # [B, H, W]

    # non-instrument student masks, re-normalised
    student = torch.stack([masks[:, c] for c in non_inst], dim=1)  # [B, K_ch, H, W]
    student = student / (student.sum(dim=1, keepdim=True) + eps)

    # angle → sector label (deterministic, independent of mask state)
    # atan2 range [-pi, pi] → mapped to [0, K_ch) uniformly
    with torch.no_grad():
        angle  = torch.atan2(flow[:, 1], flow[:, 0])                     # [B, H, W]
        sector = ((angle + math.pi) / (2 * math.pi) * K_ch).long()      # [B, H, W]
        sector = sector.clamp(0, K_ch - 1)

    total_ce  = torch.tensor(0.0, device=masks.device)
    total_div = torch.tensor(0.0, device=masks.device)
    n_pairs   = K_ch * (K_ch - 1) // 2

    for b in range(B):
        pw      = pixel_w[b]                                 # [H, W]
        w_total = pw.sum() + eps
        lmap    = sector[b]                                  # [H, W], values 0..K_ch-1

        # one-hot CE: non_inst[k] should activate where sector==k
        target = F.one_hot(lmap, K_ch).permute(2, 0, 1).float()  # [K_ch, H, W]
        log_s  = torch.log(student[b] + eps)                      # [K_ch, H, W]
        ce     = -(target * log_s).sum(dim=0)                     # [H, W]
        batch_ce = (ce * pw / w_total).sum()
        total_ce = total_ce + batch_ce

        if do_debug and b == 0:
            with torch.no_grad():
                ch_probs = []
                for k in range(K_ch):
                    mask_k = (lmap == k)
                    if mask_k.sum() > 0:
                        ch_probs.append(
                            f"ch{non_inst[k]}@sec{k}={student[b,k][mask_k].mean():.3f}"
                            f"(n={mask_k.sum().item()})"
                        )
                logger.info(
                    f"[angle_ce #{_flow_angle_ce_call_count}] "
                    f"ce={batch_ce.item():.4f}  {' '.join(ch_probs)}"
                )

        if diversity_weight > 0 and n_pairs > 0:
            flow_dir = flow[b] / (flow_mag[b].unsqueeze(0) + eps)  # [2, H, W]
            mu_list  = []
            for ci in range(K_ch):
                m  = student[b, ci] * pixel_w[b]
                w  = m.sum() + eps
                mu = torch.stack([(flow_dir[0] * m).sum() / w,
                                  (flow_dir[1] * m).sum() / w])
                mu_list.append(mu / (mu.norm() + eps))
            div_b = sum(
                (mu_list[i] * mu_list[j]).sum()
                for i in range(K_ch) for j in range(i + 1, K_ch)
            ) / n_pairs
            total_div = total_div + div_b

    return (total_ce + diversity_weight * total_div) / B


def flow_boundary_tv_loss(
    masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_channels: list,
    var_window: int = 13,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Edge-aware smoothness loss (Monodepth2 style).

    Penalise mask gradient weighted by exp(-alpha * |∇flow|):
      - At flow edges  (large |∇flow|): weight ≈ 0 → mask boundary allowed
      - In smooth flow (small |∇flow|): weight ≈ 1 → mask boundary penalised

    Loss decreases as mask boundaries migrate to flow edge locations.
    """
    B, C, H, W = masks.shape
    non_inst = [c for c in range(C) if c not in instrument_channels]

    # flow assumed already at mask resolution [B, 2, H, W]
    # edge-aware smoothness (Monodepth2, get_smooth_loss), verbatim form:
    #   |∂mask| * exp(-|∂flow|), averaged over all pixels
    grad_fx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)  # [B,1,H,W-1]
    grad_fy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)  # [B,1,H-1,W]
    wx = (-alpha * grad_fx).exp()   # ~1 smooth, ~0 at edge
    wy = (-alpha * grad_fy).exp()

    loss = torch.tensor(0.0, device=masks.device)
    for c in non_inst:
        m  = masks[:, c:c + 1]
        gx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs()
        gy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs()
        loss = loss + (gx * wx).mean() + (gy * wy).mean()

    return loss / len(non_inst)


_bilateral_ce_call_count = 0

def flow_bilateral_ce_loss(
    masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_channels: list,
    window: int = 7,
    sigma: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Flow-guided bilateral self-training CE — boundary alignment without graying.

    Two decoupled jobs:
      • "who owns this region"  ← current mask argmax (STABLE; never re-assigns
        a channel, so the graying that k-means CE caused cannot happen).
      • "where the boundary is"  ← bilateral vote: each pixel's target label is
        the flow-similarity-weighted vote of its neighbours' argmax labels.

    Mechanism:
      - Flow-smooth region: bilateral kernel ≈ 1 everywhere → votes cross freely
        → a spurious mask boundary sitting in smooth flow gets out-voted and
        erased (target becomes uniform there).
      - Flow edge: kernel → 0 across the flow discontinuity → the vote is cut
        exactly at the flow edge → the target's class boundary lands on the flow
        edge.  A mask boundary offset from the flow edge sits in a strip whose
        flow matches one side → it is voted to that side → CE drags the mask
        boundary onto the flow edge.

    Target is detached and built from argmax, so CE also pushes the mask toward
    one-hot (anti-gray), and there is no per-frame channel shuffling.
    """
    global _bilateral_ce_call_count
    _bilateral_ce_call_count += 1
    do_debug = (_bilateral_ce_call_count % 50 == 1)

    B, C, H, W = masks.shape
    non_inst = [c for c in range(C) if c not in instrument_channels]
    K = len(non_inst)
    r = window // 2

    student = torch.stack([masks[:, c] for c in non_inst], dim=1)   # [B, K, H, W]
    student = student / (student.sum(dim=1, keepdim=True) + eps)

    with torch.no_grad():
        # stable region identity = current argmax, as one-hot
        L0     = student.argmax(dim=1)                              # [B, H, W]
        onehot = F.one_hot(L0, K).permute(0, 3, 1, 2).float()      # [B, K, H, W]

        # per-image flow normalisation so sigma is scale-invariant
        flow_std = flow.flatten(2).std(dim=2).clamp(min=eps)       # [B, 2]
        flow_n   = flow / flow_std[:, :, None, None]               # [B, 2, H, W]

        oh_pad   = F.pad(onehot, (r, r, r, r), mode='replicate')
        flow_pad = F.pad(flow_n, (r, r, r, r), mode='replicate')

        votes  = torch.zeros_like(onehot)                          # [B, K, H, W]
        wsum   = torch.zeros(B, 1, H, W, device=masks.device)
        inv2s2 = 1.0 / (2.0 * sigma * sigma)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                oh_s = oh_pad[:, :, r + dy:r + dy + H, r + dx:r + dx + W]
                fl_s = flow_pad[:, :, r + dy:r + dy + H, r + dx:r + dx + W]
                fd   = ((flow_n - fl_s) ** 2).sum(dim=1, keepdim=True)  # [B, 1, H, W]
                w    = torch.exp(-fd * inv2s2)
                votes = votes + w * oh_s
                wsum  = wsum + w
        target = votes / (wsum + eps)                              # [B, K, H, W] soft label

        if do_debug:
            # how much did the bilateral vote actually move the labels?
            tgt_lbl   = target.argmax(dim=1)                       # [B, H, W]
            changed   = (tgt_lbl != L0).float().mean().item()      # frac of relabelled px
            max_prob  = student.max(dim=1).values.mean().item()    # 1=sharp, 1/K=gray
            # per-channel argmax occupancy (detect dead/dominant channels)
            occ = [(L0 == k).float().mean().item() for k in range(K)]
            logger.info(
                f"[bilateral_ce #{_bilateral_ce_call_count}] "
                f"relabelled={changed*100:.2f}%  mean_max_prob={max_prob:.3f}  "
                f"argmax_occ={[f'{o:.2f}' for o in occ]}"
            )

    logp = torch.log(student + eps)
    ce   = -(target * logp).sum(dim=1)                             # [B, H, W]
    return ce.mean()


def flow_angle_outside_loss(
    non_inst_masks: torch.Tensor,
    flow: torch.Tensor,
    instrument_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Direct outside-sector penalty for each non-instrument channel.

    non_inst_masks: [B, K_ch, H, W] — K_ch-way sub-softmax computed from raw logits
        with ch1 excluded, so gradient never touches ch1.
    Loss = weighted mean of channel k's activation on pixels OUTSIDE sector k.
    """
    B, K_ch, H, W = non_inst_masks.shape

    if instrument_mask is not None:
        non_inst_w = 1.0 - instrument_mask.detach().clamp(0, 1)
    else:
        non_inst_w = torch.ones(B, H, W, device=non_inst_masks.device)

    pixel_w = non_inst_w * flow.norm(dim=1)             # [B, H, W]

    with torch.no_grad():
        angle  = torch.atan2(flow[:, 1], flow[:, 0])   # [B, H, W]
        sector = ((angle + math.pi) / (2 * math.pi) * K_ch).long().clamp(0, K_ch - 1)

    w_total = pixel_w.sum() / B + eps
    total   = torch.tensor(0.0, device=non_inst_masks.device)
    for k in range(K_ch):
        outside = (sector != k).float()
        penalty = (non_inst_masks[:, k] * outside * pixel_w).sum() / w_total
        total   = total + penalty

    return total / K_ch


_DINO_ONLY_KWARGS = frozenset({
    'dino_arch', 'dino_patch_size', 'dino_checkpoint', 'dino_input_size', 'w_dino',
    'dino_channels',
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
        flow_cluster_ce_temperature: float = 0.3,
        flow_cluster_ce_diversity: float = 0.5,
        flow_cluster_ce_start_epoch: int = 0,
        flow_cluster_ce_max_iter: int = 300,        # k-means++ max EM iterations
        flow_cluster_ce_color_weight: float = 0.0,  # >0 enables joint flow+HSV clustering
        # ── V34 flow-angle CE loss (default off) ──────────────────────────
        w_flow_angle_ce: float = 0.0,
        flow_angle_ce_diversity: float = 0.0,
        # ── flow-angle outside penalty (direct, no softmax) ───────────────
        w_flow_angle_outside: float = 0.0,
        # ── flow boundary TV loss ─────────────────────────────────────────
        w_flow_boundary_tv: float = 0.0,
        flow_boundary_tv_var_window: int = 13,
        flow_boundary_tv_alpha: float = 1.0,
        # ── flow-guided bilateral self-training CE (boundary alignment) ───
        w_flow_bilateral_ce: float = 0.0,
        flow_bilateral_window: int = 7,
        flow_bilateral_sigma: float = 0.5,
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

        self.w_flow_cluster_ce             = w_flow_cluster_ce
        self.flow_cluster_ce_temperature   = flow_cluster_ce_temperature
        self.flow_cluster_ce_diversity     = flow_cluster_ce_diversity
        self.flow_cluster_ce_start_epoch   = flow_cluster_ce_start_epoch
        self.flow_cluster_ce_max_iter      = flow_cluster_ce_max_iter
        self.flow_cluster_ce_color_weight  = flow_cluster_ce_color_weight
        self._flow_cluster_ce_active       = (flow_cluster_ce_start_epoch == 0)

        self.w_flow_angle_ce         = w_flow_angle_ce
        self.flow_angle_ce_diversity = flow_angle_ce_diversity
        self.w_flow_angle_outside    = w_flow_angle_outside
        self.w_flow_boundary_tv          = w_flow_boundary_tv
        self.flow_boundary_tv_var_window = flow_boundary_tv_var_window
        self.flow_boundary_tv_alpha      = flow_boundary_tv_alpha
        self.w_flow_bilateral_ce         = w_flow_bilateral_ce
        self.flow_bilateral_window       = flow_bilateral_window
        self.flow_bilateral_sigma        = flow_bilateral_sigma

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
                      gt_fw_flows, gt_bw_flows, pl_masks, gaps=None):
        # super() = RCFDinoModel which:
        #   1. resets _captured_mask_logits
        #   2. runs RCFModel.forward_train (sets _captured_mask_logits, calls
        #      pred_separate_residual which sets _captured_residual_fw)
        #   3. appends L_dino
        #   4. returns WITHOUT clearing _captured_mask_logits
        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks, gaps=gaps,
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

        # 4-way sub-softmax over non-instrument channels only (ch1 truly excluded)
        _non_inst = [c for c in range(C) if c not in self.instrument_channels]
        masks0_non_inst = F.softmax(raw[:, 0][:, _non_inst], dim=1)  # [B, K, Hm, Wm]

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
            img0 = imgs[:, 0] if self.flow_cluster_ce_color_weight > 0 else None
            L, lmap_vis = flow_cluster_ce_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                K=self.flow_tv_K,
                max_iter=self.flow_cluster_ce_max_iter,
                temperature=self.flow_cluster_ce_temperature,
                instrument_mask=t_inst_mask,
                diversity_weight=self.flow_cluster_ce_diversity,
                image=img0,
                color_weight=self.flow_cluster_ce_color_weight,
            )
            losses['loss_flow_cluster_ce'] = L
            losses['loss'] = losses['loss'] + self.w_flow_cluster_ce * L

            # ── cluster visualization ─────────────────────────────────────────
            if lmap_vis is not None and (self.train_iter - 1) % self.log_interval == 0:
                try:
                    K_vis = self.flow_tv_K
                    # fixed palette: gray=dummy, then red/green/blue/yellow/cyan
                    palette = torch.tensor([
                        [0.5, 0.5, 0.5],   # cluster 0 → dummy (static)
                        [1.0, 0.2, 0.2],   # cluster 1 → ch0
                        [0.2, 1.0, 0.2],   # cluster 2 → ch2
                        [0.2, 0.2, 1.0],   # cluster 3 → ch3
                        [1.0, 1.0, 0.2],   # cluster 4 → ch4
                    ], dtype=torch.float32)                   # [K, 3]
                    colored = palette[lmap_vis]               # [H, W, 3]
                    colored = colored.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                    idx_in_batch = 0
                    img_frame_id = paths[0][idx_in_batch].split('/')[-1][:-4]
                    fn_cluster = '{}/train_iter{:07d}_{}_{}_{}_cluster.jpg'.format(
                        self.save_dir, self.train_iter - 1,
                        seq_names[idx_in_batch], seq_ids[idx_in_batch], img_frame_id,
                    )
                    torchvision.utils.save_image(colored, fn_cluster)
                except Exception as e:
                    logger.warn(f"cluster vis save failed: {e}")

        # ── 15. L_flow_angle_ce ───────────────────────────────────────────────
        if self.w_flow_angle_ce > 0:
            t_inst_mask = (teacher_p[:, self.distill_channel]
                           if teacher_p is not None else None)
            L = flow_angle_ce_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                instrument_mask=t_inst_mask,
                diversity_weight=self.flow_angle_ce_diversity,
            )
            losses['loss_flow_angle_ce'] = L
            losses['loss'] = losses['loss'] + self.w_flow_angle_ce * L

        # ── 16. L_flow_angle_outside ──────────────────────────────────────────
        if self.w_flow_angle_outside > 0:
            t_inst_mask = (teacher_p[:, self.distill_channel]
                           if teacher_p is not None else None)
            L = flow_angle_outside_loss(
                masks0_non_inst, flow=flow_r,
                instrument_mask=t_inst_mask,
            )
            losses['loss_flow_angle_outside'] = L
            losses['loss'] = losses['loss'] + self.w_flow_angle_outside * L

        # ── 17. L_flow_boundary_tv ────────────────────────────────────────────
        if self.w_flow_boundary_tv > 0:
            L = flow_boundary_tv_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                alpha=self.flow_boundary_tv_alpha,
            )
            losses['loss_flow_boundary_tv'] = L
            losses['loss'] = losses['loss'] + self.w_flow_boundary_tv * L

        # ── 18. L_flow_bilateral_ce ───────────────────────────────────────────
        if self.w_flow_bilateral_ce > 0:
            L = flow_bilateral_ce_loss(
                masks0, flow=flow_r,
                instrument_channels=self.instrument_channels,
                window=self.flow_bilateral_window,
                sigma=self.flow_bilateral_sigma,
            )
            losses['loss_flow_bilateral_ce'] = L
            losses['loss'] = losses['loss'] + self.w_flow_bilateral_ce * L

        return losses

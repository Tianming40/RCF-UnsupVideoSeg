"""
tissue_role_loss.py — auxiliary losses for channel role specialisation.

Added in V3:
  L_grasp_flow  : tissue channel aligns with flow at annotated grasp point
                  (excluding instrument region to avoid reinforcing ch1)
  L_dissect_prox: tissue channel activates near annotated dissection point
                  (Gaussian spatial prior from annotation)

V2 loss set:
  L_rigid       : instrument channels have consistent internal flow (rigid body)
  L_grasp_conv  : grasping-point channel aligns with flow convergence (divergence prior)
  L_deform      : tissue channel has larger residual than background (non-rigid deformation)
  L_align       : tissue channel flow aligns with grasping-point flow direction
  L_motion      : tissue channel moves more than background (hinge)

Removed from V1:
  L_compact     : spatial variance — caused instrument masks to fragment
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rigidity loss  (replaces L_compact)
# ─────────────────────────────────────────────────────────────────────────────

def rigidity_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        instrument_channels: list,
) -> torch.Tensor:
    """
    Instrument channels should have spatially uniform flow (rigid-body motion).
    Penalises the weighted variance of the flow vector within each instrument
    channel — does NOT compress the mask shape, only the flow variation.

    masks              : [B, C, H, W]
    flow               : [B, 2, H, W]  RAFT flow at mask resolution
    instrument_channels: list of channel indices

    Returns scalar (to minimise).
    """
    losses = []
    for c in instrument_channels:
        m = masks[:, c]                                        # [B, H, W]
        w = m.sum(dim=(1, 2)) + 1e-6                           # [B]

        # Weighted mean flow within this channel
        mu_x = (flow[:, 0] * m).sum(dim=(1, 2)) / w           # [B]
        mu_y = (flow[:, 1] * m).sum(dim=(1, 2)) / w           # [B]

        # Weighted flow variance (lower = more rigid)
        var_x = ((flow[:, 0] - mu_x[:, None, None]).pow(2) * m).sum(dim=(1, 2)) / w
        var_y = ((flow[:, 1] - mu_y[:, None, None]).pow(2) * m).sum(dim=(1, 2)) / w

        losses.append((var_x + var_y).mean())

    return sum(losses) / max(len(losses), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grasping-point convergence loss  (convergence prior)
# ─────────────────────────────────────────────────────────────────────────────

def grasping_convergence_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        grasping_channel: int,
) -> torch.Tensor:
    """
    The grasping-point channel (ch2) should overlap with the region of
    maximum flow convergence, computed from the numerical divergence of
    the RAFT flow field.

    Intuition: where -div(flow) is largest = where flow 'sinks' = instrument
    tip touching tissue. This is fully unsupervised (no grasping-point label).

    masks           : [B, C, H, W]
    flow            : [B, 2, H, W]
    grasping_channel: channel index (typically 2, highest convergence empirically)

    Returns scalar (to minimise).
    """
    # Numerical divergence via central differences (replicate padding)
    fx_p = F.pad(flow[:, 0:1], (1, 1, 1, 1), mode='replicate').squeeze(1)
    fy_p = F.pad(flow[:, 1:2], (1, 1, 1, 1), mode='replicate').squeeze(1)
    dfx_dx = (fx_p[:, 1:-1, 2:] - fx_p[:, 1:-1, :-2]) / 2.0
    dfy_dy = (fy_p[:, 2:, 1:-1] - fy_p[:, :-2, 1:-1]) / 2.0
    div = dfx_dx + dfy_dy                                       # [B, H, W]

    # Convergence = positive part of -divergence
    convergence = (-div).clamp(min=0)

    # Normalise per sample so scale doesn't depend on flow magnitude
    B = convergence.shape[0]
    max_c = convergence.view(B, -1).max(dim=1).values           # [B]
    convergence = convergence / (max_c.view(B, 1, 1) + 1e-6)   # [B, H, W]

    m_grasp = masks[:, grasping_channel]                        # [B, H, W]
    return -(m_grasp * convergence).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deformation loss
# ─────────────────────────────────────────────────────────────────────────────

def deformation_loss(
        masks: torch.Tensor,
        residual_fw: torch.Tensor,
        tissue_channel: int,
        bg_channels: tuple,
        n_classes: int,
        pred_div_coeff: float = 10.,
        residual_scale: float = 10.,
        margin: float = 0.5,
) -> torch.Tensor:
    """
    Tissue is non-rigid — its intra-segment residual flow R̂ should exceed
    the background residual by at least `margin`.

    residual_fw : [B, 2*n_classes, H, W]  raw decode_head3 output (fw direction)
    masks       : [B, C, H, W]

    Returns scalar (to minimise).
    """
    # Per-channel residual magnitude after tanh scaling
    # residual_fw → [B, 2, C, H, W] → ||·|| → [B, C, H, W]
    res = torch.tanh(residual_fw.unflatten(1, (2, n_classes)) / pred_div_coeff) \
          * residual_scale                                       # [B, 2, C, H, W]
    res_mag = res.norm(dim=1)                                    # [B, C, H, W]

    # Weighted mean residual in tissue
    mt = masks[:, tissue_channel]
    wt = mt.sum(dim=(1, 2)) + 1e-6
    res_tissue = (res_mag[:, tissue_channel] * mt).sum(dim=(1, 2)) / wt  # [B]

    # Weighted mean residual in background (average over bg channels)
    mb = sum(masks[:, c] for c in bg_channels) / max(len(bg_channels), 1)
    wb = mb.sum(dim=(1, 2)) + 1e-6
    # Use first bg channel's residual map
    res_bg = (res_mag[:, bg_channels[0]] * mb).sum(dim=(1, 2)) / wb      # [B]

    return F.relu(res_bg + margin - res_tissue).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tissue-grasping flow alignment  (kept from V1, same)
# ─────────────────────────────────────────────────────────────────────────────

def tissue_flow_alignment_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        grasping_channel: int,
        tissue_channel: int,
        min_grasp_frac: float = 0.005,
) -> torch.Tensor:
    """
    Tissue channel should activate where flow aligns with the mean flow at
    the grasping point — tissue is dragged in the same direction as the tip.

    masks         : [B, C, H, W]
    flow          : [B, 2, H, W]
    min_grasp_frac: suppress on frames where ch_grasp area < this × H×W
    """
    B, C, H, W = masks.shape

    m_grasp = masks[:, grasping_channel]
    w_grasp = m_grasp.sum(dim=(1, 2)) + 1e-6
    active  = (w_grasp > min_grasp_frac * H * W).float()

    gfx = (flow[:, 0] * m_grasp).sum(dim=(1, 2)) / w_grasp
    gfy = (flow[:, 1] * m_grasp).sum(dim=(1, 2)) / w_grasp
    gf_mag = (gfx.pow(2) + gfy.pow(2)).sqrt() + 1e-6
    gfx, gfy = gfx / gf_mag, gfy / gf_mag

    f_mag = flow.norm(dim=1, keepdim=True) + 1e-6
    fd    = flow / f_mag
    cos   = fd[:, 0] * gfx[:, None, None] + fd[:, 1] * gfy[:, None, None]

    m_tissue = masks[:, tissue_channel]
    wt = m_tissue.sum(dim=(1, 2)) + 1e-6
    loss_per = -(m_tissue * cos.clamp(min=0)).sum(dim=(1, 2)) / wt
    return (loss_per * active).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tissue motion lower bound  (kept from V1, same)
# ─────────────────────────────────────────────────────────────────────────────

def tissue_motion_loss(
        masks: torch.Tensor,
        flow_mag: torch.Tensor,
        tissue_channel: int,
        bg_channels: tuple,
        margin: float = 1.0,
        min_global_flow: float = 0.3,
) -> torch.Tensor:
    """
    Hinge: mean ||flow|| in tissue > mean ||flow|| in background + margin.
    Suppressed on static frames.
    """
    B = masks.shape[0]

    mt = masks[:, tissue_channel]
    wt = mt.sum(dim=(1, 2)) + 1e-6
    flow_tissue = (flow_mag * mt).sum(dim=(1, 2)) / wt

    mb = sum(masks[:, c] for c in bg_channels) / max(len(bg_channels), 1)
    wb = mb.sum(dim=(1, 2)) + 1e-6
    flow_bg = (flow_mag * mb).sum(dim=(1, 2)) / wb

    active = (flow_mag.view(B, -1).mean(dim=1) > min_global_flow).float()
    return (F.relu(flow_bg + margin - flow_tissue) * active).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grasp-point flow alignment  (annotation-driven, V3)
# ─────────────────────────────────────────────────────────────────────────────

def grasp_flow_alignment_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        tissue_channel: int,
        instrument_channel: int,
        grasp_xy: torch.Tensor,
        min_flow_mag: float = 0.5,
) -> torch.Tensor:
    """
    Tissue channel should activate where flow aligns with the displacement at
    the annotated grasp point — tissue being dragged moves like the instrument tip.
    Instrument region is excluded to prevent reinforcing ch1.

    masks             : [B, C, H, W]
    flow              : [B, 2, H, W]  (pixel units)
    grasp_xy          : [B, 2]  normalised (x, y) in [0,1]; (-1,-1) = no annotation
    instrument_channel: excluded from tissue weight (detached)
    min_flow_mag      : skip samples where grasp-point flow is too small
    """
    B, _, H, W = masks.shape
    losses = []

    for b in range(B):
        gx, gy = grasp_xy[b]
        if gx < 0:           # no annotation
            continue
        px = int((gx * W).clamp(0, W - 1).item())
        py = int((gy * H).clamp(0, H - 1).item())

        v = flow[b, :, py, px]                              # [2] reference vector
        v_mag = v.norm() + 1e-6
        if v_mag.item() < min_flow_mag:
            continue

        v_dir = v / v_mag                                   # unit vector [2]
        f_mag = flow[b].norm(dim=0, keepdim=True) + 1e-6   # [1, H, W]
        f_dir = flow[b] / f_mag                             # [2, H, W]
        cos   = (f_dir * v_dir[:, None, None]).sum(dim=0)   # [H, W]

        m_tissue     = masks[b, tissue_channel]
        m_instrument = masks[b, instrument_channel].detach()
        weight = m_tissue * (1.0 - m_instrument)
        w_sum  = weight.sum() + 1e-6

        losses.append(-(weight * cos.clamp(min=0)).sum() / w_sum)

    if not losses:
        return masks.sum() * 0.0
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dissection-point proximity  (annotation-driven, V3)
# ─────────────────────────────────────────────────────────────────────────────

def dissect_proximity_loss(
        masks: torch.Tensor,
        tissue_channel: int,
        dissect_xy: torch.Tensor,
        sigma: float = 0.12,
) -> torch.Tensor:
    """
    Tissue channel should activate near the annotated dissection point.
    A Gaussian heatmap centred on the annotation acts as a soft spatial label.

    masks       : [B, C, H, W]
    dissect_xy  : [B, 2]  normalised (x, y); (-1,-1) = no annotation
    sigma       : Gaussian width in normalised units (default 0.12 ≈ 12% of image)
    """
    B, _, H, W = masks.shape
    device = masks.device
    losses = []

    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    for b in range(B):
        dx, dy = dissect_xy[b]
        if dx < 0:           # no annotation
            continue

        dist2 = (grid_x - dx).pow(2) + (grid_y - dy).pow(2)
        heatmap = torch.exp(-dist2 / (2 * sigma ** 2))
        heatmap = heatmap / (heatmap.sum() + 1e-6)

        m_tissue = masks[b, tissue_channel]
        losses.append(-(m_tissue * heatmap).sum())

    if not losses:
        return masks.sum() * 0.0
    return torch.stack(losses).mean()

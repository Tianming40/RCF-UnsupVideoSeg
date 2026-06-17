"""
tissue_role_loss.py — auxiliary losses for channel role specialisation.

Added in V4 (disabled by default, w=0):
  L_contact     : tissue channel activates in the zone adjacent to instrument mask
                  (utilises known instrument channel via spatial dilation)
  L_div_tissue  : tissue channel activates where |div(RAFT_flow)| is large
                  (RAFT divergence = direct deformation signal, no network prediction)

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
        instrument_channels: list,
        n_classes: int,
        pred_div_coeff: float = 10.,
        residual_scale: float = 10.,
        # legacy kwargs kept for backward compat (ignored in agnostic mode)
        tissue_channel: int = 2,
        bg_channels: tuple = (0, 3, 4),
        margin: float = 0.5,
) -> torch.Tensor:
    """
    Channel-agnostic residual-magnitude loss.

    Rigid objects (instrument) have small R because P̂ already fits their
    motion well.  Deformable tissue needs large R to absorb the intra-segment
    shape change.

    Encourages every non-instrument channel to have a high mask-weighted
    mean ||R_c||.  Softmax competition then naturally lets the most
    deformation-consistent channel "win" the tissue region.

    residual_fw        : [B, 2*n_classes, H, W]  raw decode_head3 output
    masks              : [B, C, H, W]
    instrument_channels: list of instrument channel indices (excluded)
    """
    # Per-channel residual magnitude after tanh + scale
    # residual_fw → [B, 2, C, H, W] → ||·||₂ over dim=1 → [B, C, H, W]
    res = torch.tanh(residual_fw.unflatten(1, (2, n_classes)) / pred_div_coeff) \
          * residual_scale                                       # [B, 2, C, H, W]
    res_mag = res.norm(dim=1)                                    # [B, C, H, W]

    non_inst = [c for c in range(n_classes) if c not in instrument_channels]
    scores = []
    for c in non_inst:
        m_c = masks[:, c]
        w = m_c.sum(dim=(1, 2)) + 1e-6
        scores.append((res_mag[:, c] * m_c).sum(dim=(1, 2)) / w)  # [B]

    # Maximise average mask-weighted ||R|| over non-instrument channels
    return -torch.stack(scores, dim=1).mean(dim=1).mean()


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


# ─────────────────────────────────────────────────────────────────────────────
# 8. Contact-zone activation  (V4, instrument-guided)
# ─────────────────────────────────────────────────────────────────────────────

def contact_zone_loss(
        masks: torch.Tensor,
        instrument_channel: int,
        tissue_channels,
        dilation_r: int = 10,
) -> torch.Tensor:
    """
    Candidate tissue channels should activate in the ring-shaped zone adjacent
    to the instrument mask.  Channel-agnostic: averages over all tissue_channels
    so the network decides which channel(s) cover the contact zone.

    masks              : [B, C, H, W]
    instrument_channel : known instrument channel (detached — no gradient)
    tissue_channels    : int or list[int] — candidate non-instrument channels
    dilation_r         : dilation radius in pixels (mask-resolution pixels)

    Returns scalar (to minimise).
    """
    if isinstance(tissue_channels, int):
        tissue_channels = [tissue_channels]

    m_inst = masks[:, instrument_channel].detach().unsqueeze(1)  # [B, 1, H, W]

    k = 2 * dilation_r + 1
    m_dilated = F.max_pool2d(m_inst, kernel_size=k, stride=1, padding=dilation_r)

    contact_zone = (m_dilated - m_inst).clamp(min=0).squeeze(1)  # [B, H, W]
    B = contact_zone.shape[0]
    contact_zone = contact_zone / (
        contact_zone.view(B, -1).max(dim=1).values.view(B, 1, 1) + 1e-6
    )

    scores = []
    for c in tissue_channels:
        m_c = masks[:, c]
        w = m_c.sum(dim=(1, 2)) + 1e-6
        scores.append((m_c * contact_zone).sum(dim=(1, 2)) / w)  # [B]

    # Average over candidate channels: softmax competition decides the winner
    return -torch.stack(scores, dim=1).mean(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Flow-divergence tissue activation  (V4, RAFT-based deformation signal)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 9a. Flow-cosine assignment loss  (V8, direct RAFT-guided channel clustering)
# ─────────────────────────────────────────────────────────────────────────────

def flow_cosine_assignment_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        instrument_channels: list,
        temperature: float = 0.5,
        instrument_mask: torch.Tensor = None,
        diversity_weight: float = 0.0,
) -> torch.Tensor:
    """
    Non-instrument channels should group pixels with similar RAFT flow directions.
    Instrument pixels are excluded so their dominant motion doesn't bias the signal.

    For each non-instrument channel, compute the mask-weighted mean RAFT flow
    direction (detached, used as a fixed pseudo-label).  Then push the channel
    mask to assign high weight to pixels most cosine-similar to that mean.

    masks               : [B, C, H, W]
    flow                : [B, 2, H, W]  RAFT pre-computed flow at mask resolution
    instrument_channels : list[int]  — excluded from both channel pool and pixels
    temperature         : softmax temperature for soft-label sharpness (lower = harder)
    instrument_mask     : [B, H, W] optional — teacher's instrument prob used instead
                          of student masks; breaks feedback loop if student ch1 drifts
    diversity_weight    : weight for push-apart loss on channel mean flow directions;
                          uses non-detached mu so gradient flows back through masks
    """
    B, C, H, W = masks.shape
    non_inst = [c for c in range(C) if c not in instrument_channels]

    # suppress instrument pixels — prefer teacher mask to avoid feedback loop
    if instrument_mask is not None:
        inst_mask = instrument_mask.detach().clamp(0, 1)
    else:
        inst_mask = sum(masks[:, c] for c in instrument_channels).clamp(0, 1).detach()
    non_inst_weight = 1.0 - inst_mask                          # [B, H, W]

    # normalised flow direction + magnitude
    flow_mag = flow.norm(dim=1)                                # [B, H, W]
    flow_dir = flow / (flow_mag.unsqueeze(1) + 1e-6)          # [B, 2, H, W]

    # pixel weight: non-instrument AND moving (static pixels carry no info)
    pixel_w = non_inst_weight * flow_mag                       # [B, H, W]

    # ── mean flow direction per non-instrument channel (detached pseudo-label) ──
    mu_list = []
    for c in non_inst:
        m = (masks[:, c] * pixel_w).detach()                  # [B, H, W]
        w = m.sum(dim=(1, 2)) + 1e-6                          # [B]
        mu_x = (flow_dir[:, 0] * m).sum(dim=(1, 2)) / w      # [B]
        mu_y = (flow_dir[:, 1] * m).sum(dim=(1, 2)) / w      # [B]
        mu = torch.stack([mu_x, mu_y], dim=1)                 # [B, 2]
        mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-6)       # unit vector
        mu_list.append(mu)

    # ── cosine similarity: each pixel vs each channel's mean flow ──────────────
    cos_sims = []
    for mu in mu_list:
        cos = (flow_dir[:, 0] * mu[:, 0, None, None] +
               flow_dir[:, 1] * mu[:, 1, None, None])         # [B, H, W]
        cos_sims.append(cos)
    cos_sims = torch.stack(cos_sims, dim=1)                   # [B, K, H, W]

    # soft assignment from cosine similarity (fixed target — detached)
    target = F.softmax(cos_sims / temperature, dim=1).detach()  # [B, K, H, W]

    # student: non-instrument masks renormalised to sum=1 among themselves
    student = torch.stack([masks[:, c] for c in non_inst], dim=1)  # [B, K, H, W]
    student = student / (student.sum(dim=1, keepdim=True) + 1e-6)

    # cross-entropy, weighted by moving non-instrument pixels
    loss = -(target * torch.log(student + 1e-6)).sum(dim=1)   # [B, H, W]
    w_sum = pixel_w.view(B, -1).sum(dim=1).view(B, 1, 1) + 1e-6
    assignment_loss = (loss * pixel_w / w_sum).sum(dim=(1, 2)).mean()

    if diversity_weight <= 0 or len(non_inst) < 2:
        return assignment_loss

    # ── channel diversity: push mean flow directions apart ───────────────────
    # recompute mu WITHOUT detach so gradient flows back through masks
    mu_grad = []
    for c in non_inst:
        m = masks[:, c] * pixel_w                             # [B, H, W], no detach
        w = m.sum(dim=(1, 2)) + 1e-6                          # [B]
        mu_x = (flow_dir[:, 0] * m).sum(dim=(1, 2)) / w      # [B]
        mu_y = (flow_dir[:, 1] * m).sum(dim=(1, 2)) / w      # [B]
        mu = torch.stack([mu_x, mu_y], dim=1)                 # [B, 2]
        mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-6)
        mu_grad.append(mu)

    div_loss = torch.tensor(0.0, device=masks.device)
    n_pairs = 0
    for i in range(len(mu_grad)):
        for j in range(i + 1, len(mu_grad)):
            # cosine similarity between channel i and j mean flow (unit vectors)
            cos_ij = (mu_grad[i] * mu_grad[j]).sum(dim=1).mean()  # scalar
            div_loss = div_loss + cos_ij
            n_pairs += 1
    div_loss = div_loss / n_pairs   # mean pairwise cosine sim; minimize → push apart

    return assignment_loss + diversity_weight * div_loss


# ─────────────────────────────────────────────────────────────────────────────
# 9b. Flow-variance tissue activation  (V5, area signal)
# ─────────────────────────────────────────────────────────────────────────────

def tissue_flow_variance_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        tissue_channels,
) -> torch.Tensor:
    """
    Non-instrument channels should have HIGH internal flow variance (non-rigid).

    This is a true AREA signal: measures the heterogeneity of motion WITHIN each
    mask region, NOT spatial gradients → signal is not concentrated at boundaries.

    Rigid objects (instrument) have spatially uniform flow inside their mask
    → low mask-weighted variance.  Deformable tissue has spatially varying
    flow inside its mask → high variance.

    masks          : [B, C, H, W]
    flow           : [B, 2, H, W]  RAFT pre-computed flow at mask resolution
    tissue_channels: int or list[int]  non-instrument channel indices

    Returns scalar (to minimise, i.e. maximises variance).
    """
    if isinstance(tissue_channels, int):
        tissue_channels = [tissue_channels]

    scores = []
    for c in tissue_channels:
        m = masks[:, c]                                          # [B, H, W]
        w = m.sum(dim=(1, 2)) + 1e-6                             # [B]

        mu_x = (flow[:, 0] * m).sum(dim=(1, 2)) / w             # [B]
        mu_y = (flow[:, 1] * m).sum(dim=(1, 2)) / w             # [B]

        var_x = ((flow[:, 0] - mu_x[:, None, None]).pow(2) * m).sum(dim=(1, 2)) / w
        var_y = ((flow[:, 1] - mu_y[:, None, None]).pow(2) * m).sum(dim=(1, 2)) / w

        scores.append((var_x + var_y).mean())

    return -torch.stack(scores).mean()   # negative → minimise = maximise variance


# ─────────────────────────────────────────────────────────────────────────────
# 9c. Flow-divergence tissue activation  (V4, RAFT-based deformation signal)
# ─────────────────────────────────────────────────────────────────────────────

def tissue_divergence_loss(
        masks: torch.Tensor,
        flow: torch.Tensor,
        tissue_channels,
        instrument_channels=None,
) -> torch.Tensor:
    """
    Tissue channels should capture regions with high |div(RAFT_flow)|.

    Key design: normalise by total div energy (not by mask area).
    Score = "what fraction of the total div signal does this channel capture?"
    Gradient ∝ div²_pixel / total_div → strong, proportional to deformation.

    Also adds contrastive pressure: instrument channels should cover LOW-div
    regions, tissue channels should cover HIGH-div regions.

    masks               : [B, C, H, W]
    flow                : [B, 2, H, W]  pre-computed RAFT flow at mask resolution
    tissue_channels     : list[int] — non-instrument candidate channels
    instrument_channels : list[int] or None — if given, add contrastive term
    """
    if isinstance(tissue_channels, int):
        tissue_channels = [tissue_channels]

    fx_p = F.pad(flow[:, 0:1], (1, 1, 1, 1), mode='replicate').squeeze(1)
    fy_p = F.pad(flow[:, 1:2], (1, 1, 1, 1), mode='replicate').squeeze(1)
    dfx_dx = (fx_p[:, 1:-1, 2:] - fx_p[:, 1:-1, :-2]) / 2.0
    dfy_dy = (fy_p[:, 2:, 1:-1] - fy_p[:, :-2, 1:-1]) / 2.0
    abs_div = (dfx_dx + dfy_dy).abs()                              # [B, H, W]

    B = abs_div.shape[0]

    # Square to amplify peaks and suppress low-div noise; normalise so spatial
    # sum = 1 per image → score is "fraction of total div energy captured".
    div_sq = abs_div ** 2
    total_div = div_sq.view(B, -1).sum(dim=1).view(B, 1, 1) + 1e-6
    div_weight = div_sq / total_div                                 # [B, H, W]

    tissue_scores = []
    for c in tissue_channels:
        tissue_scores.append((masks[:, c] * div_weight).sum(dim=(1, 2)))  # [B]
    tissue_score = torch.stack(tissue_scores, dim=1).mean(dim=1).mean()

    if instrument_channels is not None:
        if isinstance(instrument_channels, int):
            instrument_channels = [instrument_channels]
        instr_scores = []
        for c in instrument_channels:
            instr_scores.append((masks[:, c] * div_weight).sum(dim=(1, 2)))
        instr_score = torch.stack(instr_scores, dim=1).mean(dim=1).mean()
        # minimise: push tissue toward div, instrument away from div
        return instr_score - tissue_score

    return -tissue_score

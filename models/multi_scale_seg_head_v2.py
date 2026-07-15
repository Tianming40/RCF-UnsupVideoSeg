"""
MultiScaleSegHeadV2
===================
Drop-in replacement for MultiScaleSegHead with three optional improvements:

  use_concat_fusion (bool, default False)
    Replace the three independent proj → element-wise sum with
    concat([proj1, proj2, proj3]) → 1×1 BN ReLU.
    Allows free cross-scale, cross-channel mixing instead of fixing
    channel-i of each scale to contribute only to channel-i of the output.

  use_aspp_attn (bool, default False, requires use_aspp=True)
    Add a NonLocalBlock immediately after the ASPP fuse step.
    Q is at full H/8 resolution; K and V are average-pooled 2× (H/16) so the
    attention matrix stays at [B, N, M] = [B, H/8·W/8, H/16·W/16].
    The output projection W_out is zero-initialised → identity at init.

  use_attention_gate (bool, default False)
    Gate the feat0 skip connection with the upsampled semantic features before
    concat.  Follows Attention U-Net (Oktay et al. 2018): spatial attention map
    [B,1,H/4,W/4] = sigmoid(W_g(fused_up) + W_x(feat0)), suppresses noisy or
    non-discriminative regions in feat0 (specularities, blood artefacts).
    No BN inside the gate (original paper convention).

  use_strip_pooling (bool, default False)
    Add a Strip Pooling module (Hou et al., CVPR 2020) after the ASPP fuse
    step. ASPP's dilated convs sample an isotropic (square) grid at every
    rate; a thin elongated structure like an instrument shaft benefits from
    long, THIN receptive fields aligned with its own axis instead. Strip
    Pooling pools along H and W independently (1×W and H×1 strips), so
    context is gathered along a full row/column without mixing in the
    perpendicular direction's irrelevant content, then fuses back additively.
    Output projection BN is zero-initialised → identity at init.

  use_dense_aspp (bool, default False)
    REPLACES plain ASPP (not stacked with it — mutually exclusive with
    use_aspp, checked at construction) with DenseASPP (Yang et al., CVPR
    2018). Plain ASPP's branches are independent and parallel; DenseASPP
    cascades dilated conv layers DenseNet-style — each layer's input is the
    concatenation of the original features and ALL previous layers' dilated
    outputs. Later (larger-rate) layers therefore see an ensemble of the
    smaller-scale context too, which both densifies the effective dilation
    coverage (fills gaps between the fixed small set of rates) and mitigates
    each individual branch's own gridding artifact. Config via
    dense_aspp_rates / dense_aspp_growth_rate.

All other parameters and behaviour are identical to MultiScaleSegHead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_scale_seg_head import _ASPPModule


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _NonLocalBlock(nn.Module):
    """
    Self-attention with pooled K/V.
      Q: full spatial resolution  [B, N, mid]   N = H*W
      K: pooled 2× resolution     [B, mid, M]   M = (H/2)*(W/2)
      V: pooled 2× resolution     [B, M, C]
    Attention matrix: [B, N, M] ≈ [B, 2304, 576] for H/W=48 → ~42 MB/batch.
    W_out BN initialised to zero so the block starts as identity.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.theta = nn.Conv2d(channels, mid, 1, bias=False)
        self.phi   = nn.Conv2d(channels, mid, 1, bias=False)
        self.g     = nn.Conv2d(channels, channels, 1, bias=False)
        self.pool  = nn.AvgPool2d(2)
        self.W_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.W_out[1].weight)
        nn.init.zeros_(self.W_out[1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_pool = self.pool(x)                                     # [B, C, H/2, W/2]

        q = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)   # [B, N, mid]
        k = self.phi(x_pool).view(B, -1, (H // 2) * (W // 2))   # [B, mid, M]
        v = self.g(x_pool).view(B, C, -1).permute(0, 2, 1)      # [B, M, C]

        attn = torch.softmax(torch.bmm(q, k), dim=-1)            # [B, N, M]
        y = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.W_out(y)


class _AttentionGate(nn.Module):
    """
    Spatial attention gate for skip connections (Attention U-Net, Oktay 2018).
      g: gate signal  — upsampled semantic features [B, g_ch, H, W]
      x: skip signal  — feat0 fine-detail features  [B, x_ch, H, W]
    Returns x * sigmoid(W_g(g) + W_x(x)), same shape as x.
    No BN in the gate projections (original paper convention).
    """

    def __init__(self, g_ch: int, x_ch: int, mid_ch: int):
        super().__init__()
        self.W_g  = nn.Conv2d(g_ch,  mid_ch, 1, bias=True)
        self.W_x  = nn.Conv2d(x_ch,  mid_ch, 1, bias=True)
        self.psi  = nn.Conv2d(mid_ch, 1,      1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        attn = torch.sigmoid(self.psi(self.relu(self.W_g(g) + self.W_x(x))))
        return x * attn


class _StripPoolingModule(nn.Module):
    """
    Strip Pooling (Hou et al., "Strip Pooling: Rethinking Spatial Pooling for
    Scene Parsing", CVPR 2020). Captures long, thin structures via 1D pooling
    along H and W separately, instead of the isotropic square context an ASPP
    branch or standard conv provides.
      horizontal strip: pool over W -> [B,C,H,1] -> 1D conv along H -> broadcast over W
      vertical strip:    pool over H -> [B,C,1,W] -> 1D conv along W -> broadcast over H
    Sum the two, fuse with a 1x1 conv (BN zero-init -> identity at init), add residual.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # -> [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # -> [B, C, 1, W]
        self.conv_h = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.conv_w = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool_h(x).squeeze(-1)                       # [B, C, H]
        h = self.conv_h(h).unsqueeze(-1).expand(-1, -1, -1, W)   # [B, C, H, W]

        w = self.pool_w(x).squeeze(-2)                       # [B, C, W]
        w = self.conv_w(w).unsqueeze(-2).expand(-1, -1, H, -1)   # [B, C, H, W]

        combined = F.relu(h + w)
        return x + self.fuse(combined)


class _DenseASPPModule(nn.Module):
    """
    DenseASPP (Yang et al., "DenseASPP for Semantic Segmentation in Street
    Scenes", CVPR 2018). Replaces ASPP's independent parallel dilated
    branches with a DenseNet-style cascade: each dilated conv layer's input
    is the concatenation of the original features and ALL previous layers'
    outputs, so later (larger-rate) layers see an ensemble of smaller-scale
    context too. This densifies the effective dilation coverage (no gaps
    between the fixed small set of rates) and mitigates each individual
    branch's own gridding artifact, without the parallel-branch redundancy
    of plain ASPP.
    Each layer: 1x1 bottleneck (-> growth_rate*2) -> 3x3 dilated conv
    (-> growth_rate), output concatenated onto the running feature stack.
    Final 1x1 conv fuses [in_ch + len(rates)*growth_rate] channels -> out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, rates=(3, 6, 12, 18), growth_rate: int = 64):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for r in rates:
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch, growth_rate * 2, 1, bias=False),
                nn.BatchNorm2d(growth_rate * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate * 2, growth_rate, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True),
            ))
            ch += growth_rate
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            y = layer(torch.cat(feats, dim=1))
            feats.append(y)
        return self.fuse(torch.cat(feats, dim=1))


class MultiScaleSegHeadV2(nn.Module):
    """
    Args: identical to MultiScaleSegHead, plus:
        use_concat_fusion  (bool): replace proj+sum with concat+1×1
        use_aspp_attn      (bool): add NonLocalBlock after ASPP (requires use_aspp=True)
        use_attention_gate (bool): gate feat0 skip connection before concat
        use_strip_pooling  (bool): add Strip Pooling module after ASPP
    """

    def __init__(
        self,
        num_classes: int = 5,
        mid_channels: int = 256,
        feat_channels=(256, 512, 1024, 2048),
        fuse_dilation: int = 6,
        use_aspp: bool = False,
        aspp_rates: tuple = (6, 12, 18),
        use_dense_aspp: bool = False,
        dense_aspp_rates: tuple = (3, 6, 12, 18),
        dense_aspp_growth_rate: int = 64,
        use_concat_fusion: bool = False,
        use_aspp_attn: bool = False,
        use_attention_gate: bool = False,
        use_strip_pooling: bool = False,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_edge_feat: bool = False,
        use_flow_feat: bool = False,
        flow_in_channels: int = 64,
        flow_drop_p: float = 0.5,
        # compat shims
        in_channels=None, in_index=None, input_transform=None,
        channels=None, norm_cfg=None, loss_decode=None,
        concat_input=None, dilation=None, num_convs=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes       = num_classes
        self.align_corners     = align_corners
        self.use_edge_feat     = use_edge_feat
        self.use_flow_feat     = use_flow_feat
        self.flow_drop_p       = flow_drop_p
        self.use_concat_fusion = use_concat_fusion
        self.use_aspp_attn     = use_aspp_attn
        self.use_attention_gate = use_attention_gate
        self.use_strip_pooling = use_strip_pooling

        if use_aspp_attn and not use_aspp:
            raise ValueError("use_aspp_attn requires use_aspp=True")
        if use_dense_aspp and use_aspp:
            raise ValueError("use_dense_aspp and use_aspp are mutually exclusive (DenseASPP replaces plain ASPP)")

        c0, c1, c2, c3 = feat_channels

        # ── 1. feature projection (feat1 / feat2 / feat3 → mid_channels) ────
        self.proj1 = _proj(c1, mid_channels)
        self.proj2 = _proj(c2, mid_channels)
        self.proj3 = _proj(c3, mid_channels)

        if use_concat_fusion:
            # concat([p1, p2, p3]) → 1×1 BN ReLU: free cross-scale channel mixing
            self.proj_fuse = _proj(mid_channels * 3, mid_channels)

        # ── 2. spatial context at H/8: dilated conv, ASPP, or DenseASPP ──────
        if use_dense_aspp:
            self.fuse_conv = _DenseASPPModule(
                mid_channels, mid_channels,
                rates=dense_aspp_rates, growth_rate=dense_aspp_growth_rate,
            )
        elif use_aspp:
            self.fuse_conv = _ASPPModule(mid_channels, mid_channels, rates=aspp_rates)
        else:
            pad = fuse_dilation
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3,
                          padding=pad, dilation=fuse_dilation, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

        if use_aspp_attn:
            self.aspp_attn = _NonLocalBlock(mid_channels)

        if use_strip_pooling:
            self.strip_pool = _StripPoolingModule(mid_channels)

        # ── 3. attention gate on feat0 skip connection ───────────────────────
        if use_attention_gate:
            c0 = feat_channels[0]
            self.attn_gate = _AttentionGate(mid_channels, c0, mid_channels // 4)

        # ── 4. optional edge enhancement ─────────────────────────────────────
        if use_edge_feat:
            self.register_buffer(
                'sobel_x',
                torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]).view(1, 1, 3, 3),
            )
            self.register_buffer(
                'sobel_y',
                torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]).view(1, 1, 3, 3),
            )
            self.edge_proj = _proj(1, mid_channels)

        # ── 5. optional flow guidance ─────────────────────────────────────────
        if use_flow_feat:
            self.flow_proj = _proj(flow_in_channels, mid_channels)

        # ── 6. decoder: upsample → (gate feat0) → concat → refine ───────────
        self.decode_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels + c0, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.decode_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # ── 6. classifier ────────────────────────────────────────────────────
        self.dropout  = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(mid_channels, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_seg.bias)
        # NonLocalBlock W_out / StripPooling fuse are zero-initialised in their
        # own __init__; re-zero here as well to survive the kaiming pass above
        # which only hits nn.Conv2d
        if self.use_aspp_attn:
            nn.init.zeros_(self.aspp_attn.W_out[1].weight)
            nn.init.zeros_(self.aspp_attn.W_out[1].bias)
        if self.use_strip_pooling:
            nn.init.zeros_(self.strip_pool.fuse[1].weight)
            nn.init.zeros_(self.strip_pool.fuse[1].bias)

    def _compute_edge(self, feat0):
        norm = feat0.norm(dim=1, keepdim=True)
        gx = F.conv2d(norm, self.sobel_x, padding=1)
        gy = F.conv2d(norm, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt()

    def forward(self, inputs, flow_feat=None):
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Step 1: project feat1/feat2/feat3 → fused [B, mid, H/8, W/8]
        if self.use_concat_fusion:
            fused = self.proj_fuse(
                torch.cat([self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)], dim=1)
            )
        else:
            fused = self.proj1(feat1) + self.proj2(feat2) + self.proj3(feat3)

        # Step 2: spatial context (dilated conv or ASPP)
        fused = self.fuse_conv(fused)

        # Step 2b: optional long-range self-attention after ASPP
        if self.use_aspp_attn:
            fused = self.aspp_attn(fused)

        # Step 2c: optional strip pooling after ASPP (long thin structure context)
        if self.use_strip_pooling:
            fused = self.strip_pool(fused)

        # Step 3: upsample to H/4
        fused = F.interpolate(
            fused, size=feat0.shape[-2:], mode='bilinear',
            align_corners=self.align_corners,
        )

        # Step 4: optional edge enhancement
        if self.use_edge_feat:
            fused = fused + self.edge_proj(self._compute_edge(feat0))

        # Step 5: optional flow guidance
        if self.use_flow_feat and flow_feat is not None:
            if (not self.training) or (torch.rand(1).item() >= self.flow_drop_p):
                fused = fused + self.flow_proj(flow_feat)

        # Step 6: (optionally gate feat0) concat, refine
        skip = self.attn_gate(fused, feat0) if self.use_attention_gate else feat0
        x = self.decode_conv1(torch.cat([fused, skip], dim=1))
        x = self.decode_conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)

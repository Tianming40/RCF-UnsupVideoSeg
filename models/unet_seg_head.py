"""
UNetSegHead
===========
Drop-in replacement for MultiScaleSegHead with explicit top-down coarse-to-fine
refinement (U-Net / FPN style).

Backbone note: with dilations=[1,1,2,4] and strides=[1,2,1,1], feat1/feat2/feat3
are all at the same spatial resolution (H/8). The refinement is therefore across
the channel/semantic hierarchy, not spatial resolution, until the final upsample
to H/4 where feat0 is concatenated.

Forward pass:
  Stage 1 — proj3(feat3)                      → x  [mid_ch, H/8]  → aux_logits[0]
  Stage 2 — x + proj2(feat2) → stage2_conv    → x  [mid_ch, H/8]  → aux_logits[1]
  Stage 3 — x + proj1(feat1) → stage3_conv(*) → x  [mid_ch, H/8]  → aux_logits[2]
  Stage 4 — upsample → concat feat0 → 2×conv  → x  [mid_ch, H/4]  → final logits

(*) stage3_conv uses dilation=fuse_dilation for wider receptive field (same as
    MultiScaleSegHead's fuse_conv), matching the original design intent.

Optional add-ons (same interface as MultiScaleSegHead):
  use_edge_feat — Sobel on feat0, injected at Stage 4 before concat
  use_flow_feat — gt flow features, injected at Stage 4 before concat

Return value:
  forward() always returns [B, num_classes, H/4, W/4]  (same as MultiScaleSegHead)

Auxiliary logits:
  After each forward call, self.last_aux_logits holds [aux1, aux2, aux3] at H/8.
  These can be used for optional auxiliary warp_seg loss in rcf_model.py; the head
  itself does not compute any loss.

Config example (decode_head2 field):
  decode_head2:
    type: UNetSegHead
    num_classes: 5
    mid_channels: 256
    feat_channels: [256, 512, 1024, 2048]
    fuse_dilation: 6
    dropout_ratio: 0.1
    align_corners: false
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Sequential:
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetSegHead(nn.Module):
    """
    Args:
        num_classes      (int)   : segmentation channels
        mid_channels     (int)   : internal feature dimension
        feat_channels    (list)  : [c0, c1, c2, c3] from backbone
        fuse_dilation    (int)   : dilation for stage3_conv (H/8 level)
        dropout_ratio    (float) : Dropout2d ratio; 0 disables
        align_corners    (bool)  : F.interpolate align_corners
        use_edge_feat    (bool)  : Sobel edge enhancement on feat0 (H/4)
        use_flow_feat    (bool)  : optical flow guidance at H/4 (training only)
        flow_in_channels (int)   : channels of incoming flow feature
        flow_drop_p      (float) : probability of dropping flow during training
    """

    def __init__(
        self,
        num_classes: int = 5,
        mid_channels: int = 256,
        feat_channels=(256, 512, 1024, 2048),
        fuse_dilation: int = 6,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_edge_feat: bool = False,
        use_flow_feat: bool = False,
        flow_in_channels: int = 64,
        flow_drop_p: float = 0.5,
        # compatibility shims for legacy config fields
        in_channels=None,
        in_index=None,
        input_transform=None,
        channels=None,
        norm_cfg=None,
        loss_decode=None,
        concat_input=None,
        dilation=None,
        num_convs=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.use_edge_feat = use_edge_feat
        self.use_flow_feat = use_flow_feat
        self.flow_drop_p = flow_drop_p

        c0, c1, c2, c3 = feat_channels
        m = mid_channels

        # ── Stage 1: feat3 only ───────────────────────────────────────────────
        self.proj3 = _proj(c3, m)
        self.stage1_conv = _conv_block(m, m)
        self.aux_seg1 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 2: += feat2 ─────────────────────────────────────────────────
        self.proj2 = _proj(c2, m)
        self.stage2_conv = _conv_block(m, m)
        self.aux_seg2 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 3: += feat1, dilated conv for wide receptive field ──────────
        self.proj1 = _proj(c1, m)
        self.stage3_conv = _conv_block(m, m, dilation=fuse_dilation)
        self.aux_seg3 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 4: upsample to H/4, concat feat0, local refinement ──────────
        self.decode_conv1 = _conv_block(m + c0, m)
        self.decode_conv2 = _conv_block(m, m)

        # ── Optional: edge enhancement (applied at H/4 before concat feat0) ───
        if use_edge_feat:
            self.sobel_x: torch.Tensor
            self.sobel_y: torch.Tensor
            self.register_buffer(
                'sobel_x',
                torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]).view(1, 1, 3, 3),
            )
            self.register_buffer(
                'sobel_y',
                torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]).view(1, 1, 3, 3),
            )
            self.edge_proj = _proj(1, m)

        # ── Optional: flow guidance (applied at H/4 before concat feat0) ──────
        if use_flow_feat:
            self.flow_proj = _proj(flow_in_channels, m)

        # ── Classifier ────────────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(m, num_classes, 1)

        # Slot for auxiliary logits — populated on every forward call
        self.last_aux_logits: List[torch.Tensor] = []

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
        for seg in [self.aux_seg1, self.aux_seg2, self.aux_seg3, self.conv_seg]:
            nn.init.normal_(seg.weight, mean=0, std=0.01)
            nn.init.zeros_(seg.bias)

    def _compute_edge(self, feat0: torch.Tensor) -> torch.Tensor:
        """Sobel magnitude on L2-norm of feat0 → [B, 1, H/4, W/4]"""
        norm = feat0.norm(dim=1, keepdim=True)
        gx = F.conv2d(norm, self.sobel_x, padding=1)
        gy = F.conv2d(norm, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt()

    def forward(self, inputs, flow_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: list of backbone features
              [0] feat0: [B, 256,  H/4, W/4]
              [1] feat1: [B, 512,  H/8, W/8]
              [2] feat2: [B, 1024, H/8, W/8]
              [3] feat3: [B, 2048, H/8, W/8]
            flow_feat: [B, flow_in_channels, H/4, W/4] or None

        Returns:
            [B, num_classes, H/4, W/4]

        Side-effect:
            self.last_aux_logits = [aux1, aux2, aux3]  at [B, num_classes, H/8, W/8]
        """
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # ── Stage 1: feat3 only ───────────────────────────────────────────────
        x = self.stage1_conv(self.proj3(feat3))        # [B, m, H/8, W/8]
        aux1 = self.aux_seg1(x)                        # [B, C, H/8, W/8]

        # ── Stage 2: += feat2 ─────────────────────────────────────────────────
        x = self.stage2_conv(x + self.proj2(feat2))    # [B, m, H/8, W/8]
        aux2 = self.aux_seg2(x)                        # [B, C, H/8, W/8]

        # ── Stage 3: += feat1, dilated conv ───────────────────────────────────
        x = self.stage3_conv(x + self.proj1(feat1))    # [B, m, H/8, W/8]
        aux3 = self.aux_seg3(x)                        # [B, C, H/8, W/8]

        self.last_aux_logits = [aux1, aux2, aux3]

        # ── Stage 4: upsample to H/4 ──────────────────────────────────────────
        x = F.interpolate(x, size=feat0.shape[-2:], mode='bilinear',
                          align_corners=self.align_corners)  # [B, m, H/4, W/4]

        # Optional edge enhancement
        if self.use_edge_feat:
            x = x + self.edge_proj(self._compute_edge(feat0))

        # Optional flow guidance (skipped entirely at val/test when flow_feat=None)
        if self.use_flow_feat and flow_feat is not None:
            use_flow = (not self.training) or (torch.rand(1).item() >= self.flow_drop_p)
            if use_flow:
                x = x + self.flow_proj(flow_feat)

        # Concat feat0 and refine
        x = self.decode_conv1(torch.cat([x, feat0], dim=1))  # [B, m, H/4, W/4]
        x = self.decode_conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)                        # [B, num_classes, H/4, W/4]

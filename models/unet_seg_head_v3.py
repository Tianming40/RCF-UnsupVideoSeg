"""
UNetSegHeadV3 — True skip-concat U-Net decoder.

Same backbone requirement as UNetSegHeadV2 (standard strides, no dilation):
  dilations: [1, 1, 1, 1]
  strides:   [1, 2, 2, 2]
  feat0 H/4 (256ch), feat1 H/8 (512ch), feat2 H/16 (1024ch), feat3 H/32 (2048ch)

Difference from V2: lateral connections use CONCAT instead of ADD.
  V2: x_up + lateral(feat)      → m channels  → stage_conv(m → m)
  V3: cat(x_up, lateral(feat))  → 2m channels → stage_conv(2m → m)

This preserves both the top-down context (x_up) and the encoder detail
(lateral) without destructive interference from addition, at the cost of
2× channels entering each stage conv (which is then squeezed back to m).

Forward:
  Stage 1: proj3(feat3)                             → stage1_conv(m→m)   → x1 [H/32]
  Stage 2: cat([upsample(x1), lateral2(feat2)])     → stage2_conv(2m→m)  → x2 [H/16]
  Stage 3: cat([upsample(x2), lateral1(feat1)])     → stage3_conv(2m→m)  → x3 [H/8]
  Stage 4: cat([upsample(x3), feat0])               → decode_conv1(m+c0→m)
                                                     → decode_conv2(m→m)
                                                     → conv_seg(m→C)      [H/4]

Aux heads at H/32, H/16, H/8 stored in self.last_aux_logits.
Interface identical to MultiScaleSegHead / UNetSegHeadV2.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetSegHeadV3(nn.Module):
    """
    Args:
        num_classes      (int)
        mid_channels     (int)   : internal feature dim (m)
        feat_channels    (list)  : [c0, c1, c2, c3] from backbone
        dropout_ratio    (float)
        align_corners    (bool)
        use_edge_feat    (bool)  : Sobel edge on feat0 injected before final concat
        use_flow_feat    (bool)  : flow guidance at H/4 (training only)
        flow_in_channels (int)
        flow_drop_p      (float)
    """

    def __init__(
        self,
        num_classes: int = 5,
        mid_channels: int = 256,
        feat_channels=(256, 512, 1024, 2048),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_edge_feat: bool = False,
        use_flow_feat: bool = False,
        flow_in_channels: int = 64,
        flow_drop_p: float = 0.5,
        # compatibility shims
        in_channels=None, in_index=None, input_transform=None,
        channels=None, norm_cfg=None, loss_decode=None,
        concat_input=None, dilation=None, num_convs=None,
        fuse_dilation=None, **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.use_edge_feat = use_edge_feat
        self.use_flow_feat = use_flow_feat
        self.flow_drop_p = flow_drop_p

        c0, c1, c2, c3 = feat_channels
        m = mid_channels

        # ── Stage 1: feat3 H/32 ───────────────────────────────────────────────
        self.proj3 = _proj(c3, m)
        self.stage1_conv = _conv_block(m, m)
        self.aux_seg1 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 2: cat(upsample(x1), lateral2(feat2))  H/16 ────────────────
        self.lateral2 = _proj(c2, m)
        self.stage2_conv = _conv_block(2 * m, m)   # 2m input (concat)
        self.aux_seg2 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 3: cat(upsample(x2), lateral1(feat1))  H/8 ─────────────────
        self.lateral1 = _proj(c1, m)
        self.stage3_conv = _conv_block(2 * m, m)   # 2m input (concat)
        self.aux_seg3 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 4: cat(upsample(x3), feat0)  H/4 ───────────────────────────
        self.decode_conv1 = _conv_block(m + c0, m)
        self.decode_conv2 = _conv_block(m, m)

        # ── Optional: Sobel edge on feat0 ─────────────────────────────────────
        if use_edge_feat:
            self.register_buffer(
                'sobel_x',
                torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]).view(1, 1, 3, 3),
            )
            self.register_buffer(
                'sobel_y',
                torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]).view(1, 1, 3, 3),
            )
            self.edge_proj = _proj(1, m)

        # ── Optional: flow guidance ────────────────────────────────────────────
        if use_flow_feat:
            self.flow_proj = _proj(flow_in_channels, m)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(m, num_classes, 1)
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
        norm = feat0.norm(dim=1, keepdim=True)
        gx = F.conv2d(norm, self.sobel_x, padding=1)
        gy = F.conv2d(norm, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt()

    def _upsample_cat(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=lateral.shape[-2:], mode='bilinear',
                             align_corners=self.align_corners)
        return torch.cat([x_up, lateral], dim=1)   # → 2m channels

    def forward(self, inputs, flow_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Stage 1
        x = self.stage1_conv(self.proj3(feat3))          # [B, m, H/32, W/32]
        aux1 = self.aux_seg1(x)

        # Stage 2 — skip concat
        x = self.stage2_conv(
            self._upsample_cat(x, self.lateral2(feat2))  # [B, 2m, H/16, W/16]
        )                                                 # [B, m,  H/16, W/16]
        aux2 = self.aux_seg2(x)

        # Stage 3 — skip concat
        x = self.stage3_conv(
            self._upsample_cat(x, self.lateral1(feat1))  # [B, 2m, H/8, W/8]
        )                                                 # [B, m,  H/8, W/8]
        aux3 = self.aux_seg3(x)

        self.last_aux_logits = [aux1, aux2, aux3]

        # Stage 4 — upsample to H/4, concat feat0 (original encoder resolution)
        x = F.interpolate(x, size=feat0.shape[-2:], mode='bilinear',
                          align_corners=self.align_corners)

        if self.use_edge_feat:
            x = x + self.edge_proj(self._compute_edge(feat0))

        if self.use_flow_feat and flow_feat is not None:
            if (not self.training) or (torch.rand(1).item() >= self.flow_drop_p):
                x = x + self.flow_proj(flow_feat)

        x = self.decode_conv1(torch.cat([x, feat0], dim=1))
        x = self.decode_conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)

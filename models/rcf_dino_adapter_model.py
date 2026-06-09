"""
rcf_dino_adapter_model.py — Domain-adapter fine-tuning on top of RCFDinoModel.

Strategy
--------
Insert lightweight residual adapters at two points of the frozen ResNet50:
  - Stage 1 output (256 ch)   → FCNHead input[0]
  - Stage 4 output (2048 ch)  → FCNHead input[1]

Each adapter is a bottleneck block (1×1 conv → BN → ReLU → 1×1 conv → BN)
with a residual connection, zero-initialised so it starts as identity.

The backbone trains at `backbone_lr_scale × base_lr` (e.g. 0.1×);
adapters + all other heads train at `base_lr`.

No existing file is modified.  The only hook used is `extract_feat()`,
a single-line method in RCFModel designed to be overridden.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .rcf_dino_model import RCFDinoModel
import utils

logger = utils.get_logger()


# ── Domain Adapter ────────────────────────────────────────────────────────── #

class DomainAdapter(nn.Module):
    """Residual bottleneck adapter: C → bottleneck → C.

    Zero-initialised last layer so the adapter outputs 0 at init,
    i.e. the full block is an identity transform initially.
    """

    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Zero-init → identity at start (residual outputs 0)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Main Model ────────────────────────────────────────────────────────────── #

class RCFDinoAdapterModel(RCFDinoModel):
    """
    RCFDinoModel + lightweight domain adapters for grasp10 fine-tuning.

    Extra __init__ kwargs (all optional):

      adapter_bottleneck_s1  int    bottleneck width for stage-1 adapter (256 ch)
                                    default: 64
      adapter_bottleneck_s4  int    bottleneck width for stage-4 adapter (2048 ch)
                                    default: 64
      backbone_lr_scale      float  backbone lr = base_lr × scale
                                    default: 0.1
    """

    def __init__(
        self,
        args,
        adapter_bottleneck_s1: int = 64,
        adapter_bottleneck_s4: int = 64,
        backbone_lr_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(args, **kwargs)

        self.backbone_lr_scale = backbone_lr_scale

        # Stage-1 adapter (256 ch) and Stage-4 adapter (2048 ch)
        self.adapter_s1 = DomainAdapter(256,  bottleneck=adapter_bottleneck_s1)
        self.adapter_s4 = DomainAdapter(2048, bottleneck=adapter_bottleneck_s4)

        s1_p = sum(p.numel() for p in self.adapter_s1.parameters()) / 1e6
        s4_p = sum(p.numel() for p in self.adapter_s4.parameters()) / 1e6
        logger.info(
            f"[Adapter] s1 256→{adapter_bottleneck_s1}→256 ({s1_p:.3f}M)  "
            f"s4 2048→{adapter_bottleneck_s4}→2048 ({s4_p:.3f}M)  "
            f"backbone_lr×{backbone_lr_scale}"
        )

    # ------------------------------------------------------------------ #
    # Apply adapters after backbone — overrides the one-liner in RCFModel #
    # ------------------------------------------------------------------ #
    def extract_feat(self, imgs: torch.Tensor, net: nn.Module):
        feats = super().extract_feat(imgs, net)
        # Only apply adapters on the main backbone (not EMA, not DINO)
        if net is self.backbone2:
            feats = list(feats)
            feats[0] = self.adapter_s1(feats[0])   # 256 ch  → FCNHead input[0]
            feats[3] = self.adapter_s4(feats[3])   # 2048 ch → FCNHead input[1]
        return feats

    # ------------------------------------------------------------------ #
    # Differential learning-rate param groups                             #
    # main.py calls this if the method exists (see configure_optimizers)  #
    # ------------------------------------------------------------------ #
    def get_param_groups(self, base_lr: float):
        """Return Adam param groups: backbone at lower lr, rest at base_lr."""

        adapter_ids  = {id(p) for p in self.adapter_s1.parameters()} | \
                       {id(p) for p in self.adapter_s4.parameters()}
        backbone_ids = {id(p) for p in self.backbone2.parameters()}

        backbone_params = [
            p for p in self.backbone2.parameters()
            if p.requires_grad and id(p) not in adapter_ids
        ]
        adapter_params = [
            p for p in list(self.adapter_s1.parameters()) +
                        list(self.adapter_s4.parameters())
            if p.requires_grad
        ]
        other_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in backbone_ids
            and id(p) not in adapter_ids
        ]

        def _mb(pl): return sum(p.numel() for p in pl) / 1e6

        logger.info(
            f"[Adapter] Param groups — "
            f"backbone {_mb(backbone_params):.2f}M @lr={base_lr * self.backbone_lr_scale:.2e}  "
            f"adapter  {_mb(adapter_params):.3f}M @lr={base_lr:.2e}  "
            f"other    {_mb(other_params):.2f}M @lr={base_lr:.2e}"
        )

        return [
            {'params': backbone_params, 'lr': base_lr * self.backbone_lr_scale},
            {'params': adapter_params,  'lr': base_lr},
            {'params': other_params,    'lr': base_lr},
        ]

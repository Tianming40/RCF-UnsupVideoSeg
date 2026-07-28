"""
DeformableCrossFrameAttentionPE: DeformableCrossFrameAttention
(models/deformable_cross_frame_attention.py) + a learned 2D position
embedding added to the QUERY feature only, before offset/weight
prediction -- discussed 260720.

Motivation: offset_weight_proj currently predicts sampling offsets purely
from the query position's own content (a 1x1 conv, translation-equivariant
-- identical content anywhere in the frame predicts the identical offset,
with zero notion of WHERE in the frame it is). Adding a position embedding
lets the offset predictor learn location-conditioned priors (e.g. this
dataset's camera framing / instrument-entry conventions), on top of its
existing content-conditioned regression.

Deliberately NOT added to value_feat (the content actually sampled via
grid_sample and fed forward into the mask branch) -- matches Deformable
DETR's own convention of only adding positional embeddings to the query
side of offset/attention-weight prediction, keeping the sampled appearance
feature itself position-agnostic.

Resolution is fixed throughout this project (train crop_size always
[384, 384] in resolution_crop_configs; eval sliding_window_size: 384) --
feat0 is 96x96 (H/4), feat1/2/3 are 48x48 (H/8) -- so a LEARNED embedding
(no extrapolation needed) was chosen over a sinusoidal one: it can fit
dataset-specific positional bias (e.g. instrument entry direction, camera
vignetting) that a fixed sinusoidal prior cannot. Implemented as separable
row/col nn.Embedding tables (DETR's own learned-PE convention) sliced to
the actual (H, W) at forward time, so it works unmodified across all 4
scales (different H, W each) and any sliding-window edge case where the
window is smaller than the configured size.

Subclasses DeformableCrossFrameAttention, overriding ONLY
_attend_one_direction to inject the position embedding into query_feat
before delegating to the parent's unmodified offset/value/sampling logic --
zero duplicated math. v124 (which uses the base class directly) is
completely unaffected.
"""
import torch
import torch.nn as nn

from models.deformable_cross_frame_attention import DeformableCrossFrameAttention


class LearnedPositionEmbedding2D(nn.Module):
    def __init__(self, channels: int, max_len: int = 128):
        super().__init__()
        half = channels // 2
        self.row_embed = nn.Embedding(max_len, half)
        self.col_embed = nn.Embedding(max_len, channels - half)
        nn.init.uniform_(self.row_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.col_embed.weight, -0.1, 0.1)

    def forward(self, H: int, W: int, device) -> torch.Tensor:
        rows = self.row_embed(torch.arange(H, device=device))  # [H, half]
        cols = self.col_embed(torch.arange(W, device=device))  # [W, channels-half]
        pos = torch.cat([
            rows.unsqueeze(1).expand(H, W, rows.shape[-1]),
            cols.unsqueeze(0).expand(H, W, cols.shape[-1]),
        ], dim=-1)                                  # [H, W, channels]
        return pos.permute(2, 0, 1).unsqueeze(0)     # [1, channels, H, W]


class DeformableCrossFrameAttentionPE(DeformableCrossFrameAttention):
    def __init__(self, channels: int, out_channels: int = 64, heads: int = 8,
                 num_points: int = 4, pos_max_len: int = 128):
        super().__init__(channels, out_channels, heads, num_points)
        self.pos_embed = LearnedPositionEmbedding2D(channels, max_len=pos_max_len)

    def _attend_one_direction(self, query_feat: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_feat.shape
        pos = self.pos_embed(H, W, query_feat.device).expand(B, -1, -1, -1)
        query_feat = query_feat + pos
        return super()._attend_one_direction(query_feat, value_feat)

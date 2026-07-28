"""
LocalCorrelationCrossFrameAttentionPE: LocalCorrelationCrossFrameAttention
(models/local_correlation_cross_frame_attention.py) + a learned 2D position
embedding added to the QUERY side only, discussed 260724 -- same pattern as
DeformableCrossFrameAttentionPE (models/deformable_cross_frame_attention_pe.py,
v130), applied to the newer correlation-based mechanism instead of the
blind-offset-regression one.

Same rationale as v130's PE (see deformable_cross_frame_attention_pe.py's
docstring for the full writeup): query_proj/key_proj/value_proj are all 1x1
convs, translation-equivariant, so identical content anywhere in the frame
produces identical query/key/value vectors regardless of position. Adding a
position embedding to the RAW query feature (before query_proj, matching
where v130 injects it -- before offset_weight_proj) lets the correlation
itself become location-conditioned -- e.g. the network can learn that a
given content pattern near the frame edge should weight candidates
differently than the same content pattern near the centre, on top of the
existing pure content-similarity signal.

Deliberately query-side only, not key/value-side: the actual content being
matched and aggregated (key/value, both derived from the OTHER frame) stays
position-agnostic -- position only informs how the query decides to weight
candidates, matching Deformable DETR's and v130's own convention.

Subclasses LocalCorrelationCrossFrameAttention, overriding ONLY
_attend_one_direction to inject the position embedding into query_feat
before delegating to the parent's unmodified correlation/softmax/aggregation
logic -- zero duplicated math, v138 itself (which uses the base class
directly) is completely unaffected.
"""
import torch

from models.local_correlation_cross_frame_attention import LocalCorrelationCrossFrameAttention
from models.deformable_cross_frame_attention_pe import LearnedPositionEmbedding2D


class LocalCorrelationCrossFrameAttentionPE(LocalCorrelationCrossFrameAttention):
    def __init__(self, channels: int, out_channels: int = 64,
                 proj_channels: int = 32, radius: int = 4, pos_max_len: int = 128):
        super().__init__(channels, out_channels, proj_channels, radius)
        self.pos_embed = LearnedPositionEmbedding2D(channels, max_len=pos_max_len)

    def _attend_one_direction(self, query_feat: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_feat.shape
        pos = self.pos_embed(H, W, query_feat.device).expand(B, -1, -1, -1)
        query_feat = query_feat + pos
        return super()._attend_one_direction(query_feat, value_feat)

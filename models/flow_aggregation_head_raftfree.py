"""
FlowAggregationHeadRaftFree: FlowAggregationHeadWithResidualV2, unchanged in
every mechanic (per-channel rigid+affine+residual motion fitting, topk
hard-case selection / LQCD, bg-affine removal, ...), except the STATIC
boundary-angle detection (HQAM, detect_flow_changes_batch: threshold the
flow-angle change between neighbouring pixels) is replaced with a DYNAMIC,
externally-supplied per-pixel confidence map.

Rationale (session discussion 260728): HQAM's boundary-only restriction
exists to protect the mask/motion-clustering loss from RAFT's own
unreliability in low-signal regions (occlusion, low light, textureless
tissue interior) -- it's a defence against a specific, EXTERNAL flow
source's known failure mode. Once RAFT is removed (see
models/local_correlation_flow_head.py, models/rcf_selftaught_flow_model.py)
and replaced by a self-taught flow trained with an edge-aware smoothness
regulariser (which explicitly propagates reliable boundary-adjacent motion
into low-texture interiors -- the same class of region HQAM was
protecting), the flow head's OWN per-pixel reconstruction error becomes a
more precise, self-diagnosing confidence signal than a hand-tuned static
angle threshold: low reconstruction error -> this pixel's flow estimate is
trustworthy -> full supervision weight; high error -> down-weight, same
spirit as HQAM but derived from evidence instead of a fixed heuristic.

topk (LQCD) is DELIBERATELY KEPT (config raises it from the historical 4 to
6, i.e. relaxed not removed) -- it protects against a different, orthogonal
failure mode (a whole frame's content being fundamentally uninterpretable:
lens obstruction, extreme motion blur, blood/smoke) that self-taught flow
cannot fix either, since no motion estimator -- learned or not -- can
recover signal that isn't in the frame pair to begin with.

set_external_weights(...) MUST be called before every forward() call when
this class is used in the RAFT-free pipeline; forward() calls
detect_flow_changes_batch() exactly twice per step (fw direction, then bw
direction, matching FlowAggregationHeadWithResidualV2.forward's fixed call
order for im_num=2) -- the two supplied weight maps are consumed in that
same order via a simple iterator, then the override quietly falls back to
the inherited (RAFT-oriented) angle-threshold detector on any call beyond
those two, so accidentally forgetting to call set_external_weights for a
given step reverts to V2's original behaviour rather than crashing (visible
in loss curves as a regression, not a silent no-op).
"""
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2


class FlowAggregationHeadRaftFree(FlowAggregationHeadWithResidualV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_weights_iter = None

    def set_external_weights(self, weight_fw, weight_bw):
        """weight_fw/bw: [B, 1, H, W] per-pixel confidence, same resolution
        as the flow passed into forward() (mask_size). Consumed once (fw
        then bw), in the same order FlowAggregationHeadWithResidualV2.forward
        calls detect_flow_changes_batch for im_num=2."""
        self._external_weights_iter = iter([weight_fw, weight_bw])

    def detect_flow_changes_batch(self, flow_data, threshold=None, dilation_size=None):
        if self._external_weights_iter is not None:
            try:
                return next(self._external_weights_iter)
            except StopIteration:
                self._external_weights_iter = None
        return super().detect_flow_changes_batch(flow_data, threshold, dilation_size)

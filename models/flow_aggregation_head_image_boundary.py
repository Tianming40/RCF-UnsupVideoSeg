"""
FlowAggregationHeadImageBoundary: FlowAggregationHeadWithResidualV2 with an
image-edge-derived boundary weight for the warp-reconstruction loss,
discussed 260730 as a targeted addition on top of v102 (this project's
all-time champion, 139.77/140.12 sum) -- NOT part of the RAFT-free/JEPA
experimental line (v140-143), a separate, much smaller, isolated change to
the proven RAFT-based recipe.

Motivation: HQAM (detect_flow_changes_batch, this project's adaptation of
Liu et al.'s Instrument.pdf) restricts/weights the flow-reconstruction loss
to regions where RAFT's OWN flow shows an angle discontinuity (>
boundary_threshold). This session's earlier bottleneck diagnosis (script/
diagnose_v102_bottleneck.py) found RAFT's own reliability strongly
correlates with mask error -- i.e. HQAM's detector inherits RAFT's
domain-transfer noise (low-light/specular endoscopic footage), not just
RAFT's flow VALUES. An image-edge-derived boundary signal doesn't depend on
RAFT at all, so it can't inherit that specific noise source, though it has
its own (different) noise sources -- see below.

Edge extraction pipeline (empirically tuned this session against 10 real
training frames, saved/edge_test_260729/ -- plain Sobel picked up substantial
blood-vessel/tissue-texture noise and specular-reflection false edges; a
naive DINO-feature-discontinuity attempt was WORSE, dominated by ViT
attention-sink artifacts unrelated to content):
  1. Un-normalize the crop-resolution image back to [0,1].
  2. Detect specular highlights (high brightness AND low saturation) and
     exclude them from edge computation -- removes reflection streaks.
  3. Self-guided-filter the grayscale image (edge-preserving smoothing) to
     suppress fine texture (blood vessels, tissue creases) while keeping
     large-scale structure (instrument silhouette) -- structure/texture
     separation, cheap approximation via the same guided-filter formula
     already used elsewhere in this project (main.py:_guided_filter) for
     eval-time mask smoothing.
  4. Downsample to mask_size, THEN Sobel (matches the exact order tested
     in the saved/edge_test_260729/guided_sobel_test.jpg comparison).
  5. Normalize per-sample to [0,1]. Deliberately NOT thresholded/binarized
     (session discussion: the continuous edge-strength map, un-thresholded,
     looked qualitatively best of the columns compared -- avoids picking an
     arbitrary quantile cutoff, and composes naturally with HQAM's own
     already-continuous boundary_floor convention).

boundary_mode:
  'image_edge': REPLACES HQAM's flow-angle detection entirely -- the warp
    loss is weighted purely by image-edge strength. Trades away coverage of
    motion boundaries that are real but visually inconspicuous (e.g. subtle
    tissue-deformation edges with no strong photometric contrast) in
    exchange for never inheriting RAFT's own angle-detection noise.
  'union': boundary weight = max(image-edge strength, HQAM's flow-angle
    detection) -- keeps HQAM's coverage of motion-only boundaries AND adds
    image-edge coverage wherever RAFT's angle detection is weak/noisy but
    the object boundary is visually clear. Superset of both mechanisms'
    strengths (and, unavoidably, both mechanisms' remaining noise sources).

Injection mechanism: reuses the exact same set_external_weights /
detect_flow_changes_batch-override pattern as FlowAggregationHeadRaftFree
(models/flow_aggregation_head_raftfree.py) -- NOT a subclass of it (that
class's own docstring and naming is specifically about the RAFT-free line;
this one is RAFT-based, v102's own decode_head, so a fresh small copy of
the same mechanism keeps the two lines conceptually and semantically
separate even though the plumbing is identical). models/rcf_model.py's
forward_train calls compute_image_edge_weight/detect_flow_changes_batch/
set_external_weights via a hasattr guard (see that file) -- a no-op for
every other decode_head type, so v102 itself and every other existing
config is completely unaffected.
"""
import torch
import torch.nn.functional as F

from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2


class FlowAggregationHeadImageBoundary(FlowAggregationHeadWithResidualV2):
    def __init__(self, *args, boundary_mode: str = 'union',
                 edge_guided_filter_r: int = 8,
                 edge_guided_filter_eps: float = 0.02,
                 edge_specular_brightness_th: float = 0.80,
                 edge_specular_saturation_th: float = 0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert boundary_mode in ('image_edge', 'union'), boundary_mode
        self.boundary_mode = boundary_mode
        self.edge_guided_filter_r = edge_guided_filter_r
        self.edge_guided_filter_eps = edge_guided_filter_eps
        self.edge_specular_brightness_th = edge_specular_brightness_th
        self.edge_specular_saturation_th = edge_specular_saturation_th

        self.register_buffer(
            'sobel_x_ie',
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer(
            'sobel_y_ie',
            torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('imagenet_mean_ie', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std_ie', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._external_weights_iter = None

    # ------------------------------------------------------------------ #
    # External-weight injection (same pattern as FlowAggregationHeadRaftFree,
    # kept as an independent copy -- see module docstring)               #
    # ------------------------------------------------------------------ #
    def set_external_weights(self, weight_fw, weight_bw):
        self._external_weights_iter = iter([weight_fw, weight_bw])

    def detect_flow_changes_batch(self, flow_data, threshold=None, dilation_size=None):
        if self._external_weights_iter is not None:
            try:
                return next(self._external_weights_iter)
            except StopIteration:
                self._external_weights_iter = None
        return super().detect_flow_changes_batch(flow_data, threshold, dilation_size)

    # ------------------------------------------------------------------ #
    # Image-edge weight computation                                       #
    # ------------------------------------------------------------------ #
    def _guided_filter(self, guide, src):
        r = self.edge_guided_filter_r
        eps = self.edge_guided_filter_eps

        def mean_f(x):
            return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='reflect'), 2 * r + 1, stride=1, padding=0)

        mean_I = mean_f(guide)
        mean_p = mean_f(src)
        cov_Ip = mean_f(guide * src) - mean_I * mean_p
        var_I = mean_f(guide * guide) - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return mean_f(a) * guide + mean_f(b)

    def compute_image_edge_weight(self, img_frame):
        """
        img_frame: [B, 3, H, W], ImageNet-mean/std-normalized crop-resolution
        image (the SAME tensor convention as imgs[:, i] elsewhere in this
        codebase). Returns [B, 1, mask_h, mask_w], continuous in [0, 1],
        NOT thresholded (see module docstring).
        """
        img01 = (img_frame * self.imagenet_std_ie + self.imagenet_mean_ie).clamp(0, 1)

        gray = img01.mean(dim=1, keepdim=True)
        mx = img01.amax(dim=1, keepdim=True)
        mn = img01.amin(dim=1, keepdim=True)
        sat = (mx - mn) / (mx + 1e-6)
        specular = ((gray > self.edge_specular_brightness_th) &
                    (sat < self.edge_specular_saturation_th)).float()

        gray_smooth = self._guided_filter(gray, gray)
        gray_smooth_ds = F.interpolate(gray_smooth, size=self.mask_size, mode='bilinear', align_corners=False)
        specular_ds = F.interpolate(specular, size=self.mask_size, mode='nearest')

        gx = F.conv2d(gray_smooth_ds, self.sobel_x_ie, padding=1)
        gy = F.conv2d(gray_smooth_ds, self.sobel_y_ie, padding=1)
        edge = (gx ** 2 + gy ** 2 + 1e-6).sqrt()
        edge = edge * (1.0 - specular_ds)

        edge_norm = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return edge_norm

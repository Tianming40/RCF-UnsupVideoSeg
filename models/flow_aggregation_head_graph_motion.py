"""
FlowAggregationHeadGraphMotion: FlowAggregationHeadWithResidualV2 with a
frozen-DINO graph-partitioning confidence weight injected directly into the
weighted least-squares motion fit, discussed 260730 as a third, distinct way
to bring the graph-partitioning signal into this project's motion-
reconstruction mechanism -- separate from v146 (DinoGraphFusionHead, fuses
into decode_head2's INPUT feat0) and v147 (DinoGraphEStepFusion, fuses into
decode_head2's OUTPUT mask before decode_head consumes it).

Motivation ("让Graph参与Least Squares", discussed 260730): v102's per-channel
rigid/affine motion parameters are fit by a WEIGHTED least squares
(_demean_affine_flow_per_channel in flow_aggregation_head_with_residual_v2.py),
weighted by the mask itself (mask1_agg/mask2_agg, after the base class's
_compute_agg_masks -- see that file's 260730 refactor, which pulled this
computation out of forward() into its own overridable method specifically to
support this addition without touching forward()'s other ~150 lines). This
class multiplies an additional graph-derived confidence weight into
mask1_agg/mask2_agg, directly extending the CORE motion-fitting mechanism
itself -- not a parallel signal, not an input-side or output-side addition.

GraphWeight derivation: reuses the SAME self-referential centroid-matching
math as DinoGraphEStepFusion (v147, models/dino_graph_estep_fusion.py) --
for each mask channel k, compute the weighted-average DINO-eigenvector-space
location of the pixels the CNN currently assigns to k (mu_k, weighted by the
CNN's own current mask), then P_Graph(pixel,k) = softmax over how close this
pixel is to every mu_k. But instead of fusing P_Graph back into the mask
(v147's job), this takes GraphWeight(pixel) = max_k P_Graph(pixel,k) -- the
graph's own top-1 confidence, i.e. "how decisively does the graph agree with
SOME single class for this pixel" (low when the pixel is similarly close to
multiple channels' centroids -- an ambiguous, boundary-like pixel; high when
it's clearly close to just one). Per-sample min-max normalized to [0,1], then
floor-clamped (motion_weight_floor) to avoid ever driving a pixel's fit
contribution to exactly zero -- _demean_affine_flow_per_channel divides by
mask.sum(dim=(2,3)), so a channel whose total weight collapses to ~0 for a
whole sample would make that division numerically unstable; the floor bounds
how far this mechanism alone could push it.

*** Isolation note: this file intentionally builds its OWN separate frozen
DINO instance and duplicates the eigenvector-extraction code (DINO forward /
affinity / Laplacian / eigh) from DinoGraphEStepFusion, rather than sharing
either models/dino_graph_fusion.py's or models/dino_graph_estep_fusion.py's
DINO/extraction code. Those two files are used by v146 (job 518) and v147
(job 519), both queued at the time this was written -- this class must not
touch them or the code paths they depend on in any way. This class also does
NOT share models/rcf_dino_model.py's self.dino (decode_head is constructed
independently of RCFDinoModel, before RCFDinoModel.__init__ has built its
own DINO instance, so there is no clean way to share it without invasive
cross-wiring) -- the extra frozen ViT-S/8 copy costs ~85MB, an acceptable
isolation trade-off. ***

Injection mechanism: reuses the exact same external-weight-injection pattern
already established by FlowAggregationHeadRaftFree/FlowAggregationHeadImageBoundary,
but targets the NEWLY-extracted _compute_agg_masks hook (see
flow_aggregation_head_with_residual_v2.py's 260730 refactor) instead of
detect_flow_changes_batch -- a completely separate injection slot,
set_agg_mask_weights/_agg_weights_iter, distinctly named so there is no
possible confusion with FlowAggregationHeadImageBoundary's
set_external_weights even though the two are never active in the same config.
models/rcf_model.py's forward_train calls compute_graph_motion_weight /
set_agg_mask_weights via a hasattr guard -- a no-op for every other
decode_head type, so v102, v146, v147, v144/v145 and every other existing
config is completely unaffected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2


class FlowAggregationHeadGraphMotion(FlowAggregationHeadWithResidualV2):
    def __init__(self, *args,
                 dino_arch: str = "vit_small",
                 dino_patch_size: int = 8,
                 dino_checkpoint: str = None,
                 motion_weight_input_size: int = 384,
                 motion_weight_grid_size: int = 32,
                 motion_weight_num_eigvecs: int = 10,
                 motion_weight_chunk_size: int = 8,
                 motion_weight_temperature: float = 0.1,
                 motion_weight_floor: float = 0.2,
                 **kwargs):
        super().__init__(*args, **kwargs)

        from models.rcf_dino_model import _build_dino, _FrozenModule
        raw_dino = _build_dino(dino_arch, dino_patch_size, dino_checkpoint)
        self.dino = _FrozenModule(raw_dino)
        self.dino_patch_size = dino_patch_size

        self.motion_weight_input_size = motion_weight_input_size
        self.motion_weight_grid_size = motion_weight_grid_size
        self.motion_weight_num_eigvecs = motion_weight_num_eigvecs
        self.motion_weight_chunk_size = motion_weight_chunk_size
        self.motion_weight_temperature = motion_weight_temperature
        self.motion_weight_floor = motion_weight_floor

        self._agg_weights_iter = None

    # ------------------------------------------------------------------ #
    # External-weight injection (same pattern as FlowAggregationHeadRaftFree /
    # FlowAggregationHeadImageBoundary, but a distinct slot -- see module
    # docstring) targeting _compute_agg_masks, not detect_flow_changes_batch.
    # ------------------------------------------------------------------ #
    def set_agg_mask_weights(self, weight_fw, weight_bw):
        self._agg_weights_iter = iter([weight_fw, weight_bw])

    def _compute_agg_masks(self, mask1, mask2, gt_fw_flow, gt_bw_flow, seq_names=None):
        mask1_agg, mask2_agg = super()._compute_agg_masks(
            mask1, mask2, gt_fw_flow, gt_bw_flow, seq_names=seq_names)
        if self._agg_weights_iter is not None:
            try:
                w1 = next(self._agg_weights_iter)
                w2 = next(self._agg_weights_iter)
                mask1_agg = mask1_agg * w1
                mask2_agg = mask2_agg * w2
            except StopIteration:
                self._agg_weights_iter = None
        return mask1_agg, mask2_agg

    # ------------------------------------------------------------------ #
    # Graph confidence weight computation                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_eigvecs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Deliberately duplicated from DinoGraphEStepFusion -- see module
        docstring's isolation note. imgs: [N,3,H,W] -> [N, g, G, G]."""
        N_total = imgs.shape[0]
        if self.motion_weight_chunk_size is not None and N_total > self.motion_weight_chunk_size:
            return torch.cat([
                self._compute_eigvecs_chunk(imgs[start:start + self.motion_weight_chunk_size])
                for start in range(0, N_total, self.motion_weight_chunk_size)
            ], dim=0)
        return self._compute_eigvecs_chunk(imgs)

    def _compute_eigvecs_chunk(self, imgs: torch.Tensor) -> torch.Tensor:
        S = self.motion_weight_input_size
        if imgs.shape[-2] != S or imgs.shape[-1] != S:
            imgs_r = F.interpolate(imgs, (S, S), mode='bilinear', align_corners=False)
        else:
            imgs_r = imgs

        out = self.dino(imgs_r)                      # [N, 1+P, D]
        patch = out[:, 1:]
        Hp = Wp = S // self.dino_patch_size
        D = patch.shape[-1]
        feat = patch.view(-1, Hp, Wp, D).permute(0, 3, 1, 2)   # [N, D, Hp, Wp]

        G = self.motion_weight_grid_size
        feat = F.interpolate(feat, size=(G, G), mode='bilinear', align_corners=False)
        feat = F.normalize(feat, dim=1)

        N = feat.shape[0]
        f = feat.flatten(2).transpose(1, 2)           # [N, G*G, D]
        Wm = torch.bmm(f, f.transpose(1, 2))           # [N, G*G, G*G] cosine sim
        Wm = Wm.clamp(min=0)
        eye = torch.eye(G * G, device=feat.device, dtype=feat.dtype).unsqueeze(0)
        Wm = Wm * (1 - eye)                            # zero diagonal
        deg = Wm.sum(dim=2)
        d_inv_sqrt = deg.clamp(min=1e-6).pow(-0.5)
        L = eye - d_inv_sqrt.unsqueeze(2) * Wm * d_inv_sqrt.unsqueeze(1)
        L = (L + L.transpose(1, 2)) / 2

        evals, evecs = torch.linalg.eigh(L)            # evecs: [N, G*G, G*G], ascending
        g = self.motion_weight_num_eigvecs
        v = evecs[:, :, 1:1 + g]                       # drop trivial 0th eigenvector
        v = v.transpose(1, 2).reshape(N, g, G, G)
        return v

    def compute_graph_motion_weight(self, img_frame: torch.Tensor, p_cnn_frame: torch.Tensor) -> torch.Tensor:
        """
        img_frame:    [B, 3, H, W]    ImageNet-normalized crop (imgs[:, i]).
        p_cnn_frame:  [B, K, Hm, Wm]  CNN's own softmax mask for this frame
                      (NOT detached -- gradient flows through it into the
                      CNN, same design choice as DinoGraphEStepFusion/v147).
        Returns: [B, 1, Hm, Wm], in [motion_weight_floor, 1] -- multiplied
                 directly into mask1_agg/mask2_agg, i.e. INTO the weighted
                 least-squares motion fit itself.
        """
        eigvecs = self._compute_eigvecs(img_frame)                                    # [B, g, G, G], no grad, frozen
        eigvecs = F.interpolate(eigvecs, size=p_cnn_frame.shape[-2:], mode='bilinear', align_corners=False)
        eigvecs = eigvecs.detach()

        B, K, Hm, Wm = p_cnn_frame.shape
        g = eigvecs.shape[1]
        ev_flat = eigvecs.reshape(B, g, Hm * Wm)                # [B, g, P]
        p_flat = p_cnn_frame.reshape(B, K, Hm * Wm)             # [B, K, P]

        weight_sum = p_flat.sum(dim=2, keepdim=True).clamp(min=1e-6)                  # [B, K, 1]
        mu = torch.bmm(p_flat, ev_flat.transpose(1, 2)) / weight_sum                  # [B, K, g]

        ev_flat_t = ev_flat.transpose(1, 2)                                           # [B, P, g]
        dist2 = (ev_flat_t.unsqueeze(2) - mu.unsqueeze(1)).pow(2).sum(dim=3)          # [B, P, K]
        dist2_scale = dist2.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        p_graph = F.softmax(-dist2 / dist2_scale / self.motion_weight_temperature, dim=2)  # [B, P, K]
        confidence = p_graph.max(dim=2).values.reshape(B, 1, Hm, Wm)                  # [B, 1, Hm, Wm]

        c_min = confidence.amin(dim=(2, 3), keepdim=True)
        c_max = confidence.amax(dim=(2, 3), keepdim=True)
        confidence_norm = (confidence - c_min) / (c_max - c_min + 1e-6)

        floor = self.motion_weight_floor
        return floor + (1.0 - floor) * confidence_norm

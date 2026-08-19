"""
DinoGraphBayesianPrior -- frozen-DINO graph-partitioning eigenvectors turned
into a genuine categorical PRIOR over the mask's C channels, added as a LOGIT
directly to decode_head2's raw mask logits (before the final softmax),
discussed 260811 ("先验/后验" framing).

Why this is a different mechanism from DinoGraphFusionHead (v146/v149/v152's
existing concat-into-feat3 fusion), not just another variant of it: that
mechanism concatenates eigenvectors into a mid-network CNN feature (feat0/
feat3), lets several more conv/ASPP layers reprocess them, and only then
(implicitly, opaquely) influences the final mask -- the eigenvector signal's
contribution to the final decision is whatever the CNN's later layers happen
to have learned to do with it, entangled with everything else those layers
also do. This module instead computes a genuine 5-way (mask_layer-way)
categorical distribution DIRECTLY from the eigenvectors alone, and adds its
log-probability straight onto decode_head2's own raw logits at the DECISION
layer -- i.e. exactly the additive form of Bayes' rule in log-space:
    log P(c | appearance, DINO) = log P_cnn(c | appearance)
                                 + log P_dino_prior(c | DINO eigenvectors)
(the "+ log-likelihood from motion" half of the same equation is the
project's existing-but-unused `_em_consistency_loss` -- see that method's
docstring in flow_aggregation_head_with_residual_v2.py -- kept as a SEPARATE
auxiliary loss rather than folded into this same forward pass, because the
motion likelihood needs a per-channel rigid-motion fit that itself depends
on the CURRENT mask, a circular dependency EM normally resolves by
iterating; this project already resolves it the same way _em_consistency_loss
does, as a soft-target distillation loss rather than a literal in-pass
E-step recompute).

Paper grounding (2408.14789v3, "Revisiting Surgical Instrument Segmentation
Without Human Intervention: A Graph Partitioning View"), read in full 260811
-- IMPORTANT correction to this project's earlier informal description of
"eigenvectors as semantic features": the paper's own eigenvectors are NOT a
per-pixel categorical distribution. Each eigenvector v_i (ascending
eigenvalue order) is a single real-valued partition "mode" at a specific
normalized-cut granularity -- v1/v2 (Fiedler vector) give the coarsest,
most salient split (e.g. cleanly separates instrument from background),
v3/v4 progressively finer sub-structure (e.g. specular-highlight vs real
tissue), and eigenvectors past roughly g~10-15 (for binary/part-level
granularity, per the paper's own Table 6 ablation) become "unintuitive and
noise-like". The paper's own recipe for turning this into an actual
segmentation is NOT to read off channel membership from the raw values --
it stacks the first g eigenvectors into a g-dim per-pixel EMBEDDING and runs
K-MEANS on that embedding (Sec 3.5). This module follows that recipe
directly instead of learning an opaque conv projection: soft distance to C
learnable centroids in the g-dim eigenvector-embedding space, temperature-
scaled softmax -- the exact same mathematical form as the motion side's
softmax(-error/T) likelihood, so prior and likelihood are dimensionally and
functionally symmetric before being summed.

Isolation note: eigenvector extraction is duplicated from
DinoGraphFusionHead/DinoGraphEigvecExtractor rather than shared, per this
project's established discipline of not risking any other in-flight/
completed config by refactoring a shared base.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoGraphBayesianPrior(nn.Module):
    def __init__(self, dino, dino_patch_size, num_classes,
                 dino_input_size: int = 384, grid_size: int = 32,
                 num_eigvecs: int = 10, prior_temperature: float = 1.0,
                 chunk_size: int = 8):
        super().__init__()
        self.dino = dino  # frozen _FrozenModule, shared with RCFDinoModel.dino
        self.dino_patch_size = dino_patch_size
        self.dino_input_size = dino_input_size
        self.grid_size = grid_size
        self.num_eigvecs = num_eigvecs
        self.chunk_size = chunk_size
        self.prior_temperature = prior_temperature

        # C learnable centroids in the g-dim eigenvector-embedding space --
        # this is the differentiable stand-in for the paper's K-Means step
        # (gradient-trainable centroids instead of a separate offline
        # clustering pass, so they can be learned end-to-end alongside
        # everything else via the same warp_seg loss the CNN mask is
        # trained with).
        self.centroids = nn.Parameter(torch.randn(num_classes, num_eigvecs) * 0.1)

        # zero-init output scale: prior_logit = out_scale * raw_prior_logit.
        # out_scale starts at exactly 0 -> this module's contribution to the
        # mask logits is exactly 0 at init, regardless of centroids/eigvecs
        # -> byte-identical to v149's own CNN-only mask at step 0, same
        # "exact no-op at init" discipline as every other DINO-graph module
        # this project has added. Distinct from DinoGraphFusionHead's
        # zero-init (a zero-init CONV layer) -- here it's a single scalar,
        # since the "layer" producing the logit (distance-to-centroid) has
        # no natural all-zero-weight configuration that's still a valid
        # distance metric.
        self.out_scale = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _compute_eigvecs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Identical extraction pipeline to DinoGraphFusionHead._compute_eigvecs
        (duplicated per isolation discipline, see module docstring).
        Returns [N, g, G, G]."""
        N_total = imgs.shape[0]
        if self.chunk_size is not None and N_total > self.chunk_size:
            return torch.cat([
                self._compute_eigvecs_chunk(imgs[start:start + self.chunk_size])
                for start in range(0, N_total, self.chunk_size)
            ], dim=0)
        return self._compute_eigvecs_chunk(imgs)

    def _compute_eigvecs_chunk(self, imgs: torch.Tensor) -> torch.Tensor:
        S = self.dino_input_size
        if imgs.shape[-2] != S or imgs.shape[-1] != S:
            imgs_r = F.interpolate(imgs, (S, S), mode='bilinear', align_corners=False)
        else:
            imgs_r = imgs

        out = self.dino(imgs_r)                      # [N, 1+P, D]
        patch = out[:, 1:]
        Hp = Wp = S // self.dino_patch_size
        D = patch.shape[-1]
        feat = patch.view(-1, Hp, Wp, D).permute(0, 3, 1, 2)   # [N, D, Hp, Wp]

        G = self.grid_size
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
        g = self.num_eigvecs
        v = evecs[:, :, 1:1 + g]                       # drop trivial 0th eigenvector
        v = v.transpose(1, 2).reshape(N, g, G, G)
        return v

    def forward(self, imgs: torch.Tensor, target_size) -> torch.Tensor:
        """
        imgs: [N, 3, H, W] -- same batch/order as the mask logits this will
              be added to (e.g. img_3, both frames of a pair stacked into
              the batch dim).
        target_size: (H_out, W_out) -- the mask logit resolution to upsample
              the eigenvector embedding to BEFORE the distance/softmax
              computation (upsampling the embedding, not the logits, keeps
              the centroid-distance computation meaningful at every output
              pixel rather than computing it at the coarse 32x32 grid and
              upsampling discrete-ish logits).
        Returns: [N, num_classes, H_out, W_out] -- additive log-prior,
                 == 0 everywhere at initialization (out_scale=0).
        """
        eigvecs = self._compute_eigvecs(imgs)                                   # [N, g, G, G], no grad
        eigvecs = F.interpolate(eigvecs, size=target_size, mode='bilinear', align_corners=False)
        eigvecs = eigvecs.detach()                                              # [N, g, H_out, W_out]

        # per-pixel squared distance to each of the C centroids in g-dim space
        # eigvecs: [N, g, H, W] -> [N, 1, g, H, W]; centroids: [C, g] -> [1, C, g, 1, 1]
        diff = eigvecs.unsqueeze(1) - self.centroids.view(1, -1, self.num_eigvecs, 1, 1)
        dist2 = diff.pow(2).sum(dim=2)                                          # [N, C, H, W]

        raw_prior_logit = -dist2 / max(self.prior_temperature, 1e-6)
        return self.out_scale * raw_prior_logit

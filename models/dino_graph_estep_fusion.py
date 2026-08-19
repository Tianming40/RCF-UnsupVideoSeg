"""
DinoGraphEStepFusion -- non-parametric E-step-style fusion of frozen-DINO
graph-partitioning eigenvectors with the CNN's own current soft mask,
discussed 260730 as an alternative to DinoGraphFusionHead (models/
dino_graph_fusion.py, v146) that adds NO new trainable weights.

*** Isolation note: this file intentionally DUPLICATES DinoGraphFusionHead's
DINO-forward / affinity / Laplacian / eigh extraction code (~30 lines) rather
than refactoring it into a shared helper. This is deliberate: v146 (job 518,
queued at the time this was written) must not be touched by this addition in
any way. This file, models/rcf_dino_model.py's new use_dino_graph_estep_fusion
toggle, and models/rcf_model.py's new dino_graph_estep_fusion_head hook are
all purely additive -- v146's own attribute (dino_graph_fusion_head) and code
path are untouched. ***

Motivation (discussed 260730): DINO's eigenvectors and the CNN's mask channels
live in two completely unrelated representations -- there is no reason
eigenvector index k should correspond to mask channel k. The model has no
fixed channel semantics anywhere, not even at eval time: main_tissue.py's
per-frame greedy-union oracle (validation_step, "object_channel = per-frame
greedy union oracle") only resolves channel identity AFTER the fact, using
ground truth, which is unavailable during forward_train. Any fusion that
assumes eigenvector k <-> channel k (a learned per-channel projection, or a
raw softmax over g=num_classes eigenvectors treated as class logits) risks
fusing arbitrarily-misaligned signals with no principled correspondence.

This module sidesteps that by defining "channel k" in DINO-embedding space
FROM the CNN's own current prediction, fresh every forward pass, with no
learned parameters connecting the two models at all:

  1. mu_k = weighted average, over all pixels, of the DINO eigenvector
     embedding, weighted by P_CNN(pixel, k) -- i.e. "what does DINO-space
     look like, averaged over the pixels the CNN currently assigns to
     channel k". The only thing connecting DINO's space to the CNN's channel
     k is pixel LOCATION (both are defined over the same spatial grid of the
     same frame) -- no shared feature space or learned mapping is required.
  2. P_Graph(pixel, k) = softmax_k( -||eigvec(pixel) - mu_k||^2 / temperature )
     -- how close this pixel's DINO embedding is to each channel's own
     (self-defined) centroid.
  3. P_fused(pixel, k) ∝ P_CNN(pixel, k) * P_Graph(pixel, k)^alpha, renormalized
     over k.

One step of soft clustering in DINO-embedding space, initialized by
projecting the CNN's own current soft partition into that space -- connects
to the "RCF is like EM" framing discussed earlier: mu_k is the M-step
estimate, P_Graph is the re-estimated E-step, done fresh every forward pass.

Unlike DinoGraphFusionHead (v146), there is no trainable component here, so
there is no zero-init no-op guarantee built from a learned weight -- instead
`alpha` (blend exponent, default 1.0 = the literal P_CNN*P_Graph diagram) is
the safety dial: alpha=0 makes P_Graph^0 == 1 everywhere, so P_fused reduces
to P_CNN exactly (mathematically exact no-op, not just approximately). Every
step from DINO forward through eigh and the centroid/distance computation
runs under torch.no_grad() w.r.t. DINO's own weights (frozen, never updated).
p_cnn itself is NOT detached, so gradient DOES flow from P_fused back into the
CNN through the P_CNN(pixel,k) factor and through mu_k's dependence on
p_cnn -- this is what lets loss_warp_seg's gradient teach the CNN to produce
masks that are more graph-consistent, not just a frozen post-processing step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoGraphEStepFusion(nn.Module):
    def __init__(self, dino, dino_patch_size,
                 dino_input_size: int = 384, grid_size: int = 32,
                 num_eigvecs: int = 10, chunk_size: int = 8,
                 temperature: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.dino = dino  # frozen _FrozenModule, shared with RCFDinoModel.dino
        self.dino_patch_size = dino_patch_size
        self.dino_input_size = dino_input_size
        self.grid_size = grid_size
        self.num_eigvecs = num_eigvecs
        self.chunk_size = chunk_size
        self.temperature = temperature
        # alpha in [0, 1]: P_fused ∝ P_CNN * P_Graph^alpha.
        # alpha=0 -> P_Graph^0 == 1 everywhere -> P_fused == P_CNN exactly
        #            (mathematically exact no-op).
        # alpha=1 -> the literal P(z=k) ∝ P_CNN * P_Graph fusion.
        self.alpha = alpha

    @torch.no_grad()
    def _compute_eigvecs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Deliberately duplicated from DinoGraphFusionHead -- see module
        docstring's isolation note. imgs: [N,3,H,W] -> [N, g, G, G]."""
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

    def forward(self, imgs: torch.Tensor, p_cnn: torch.Tensor) -> torch.Tensor:
        """
        imgs:  [N, 3, H, W]   -- same batch/order as p_cnn (e.g. img_3).
        p_cnn: [N, K, Hm, Wm] -- CNN's own mask, already softmax-normalized
               over K along dim=1.
        Returns: [N, K, Hm, Wm], softmax-normalized fused mask.
        """
        eigvecs = self._compute_eigvecs(imgs)                                        # [N, g, G, G], no grad, frozen
        eigvecs = F.interpolate(eigvecs, size=p_cnn.shape[-2:], mode='bilinear', align_corners=False)
        eigvecs = eigvecs.detach()                                                    # frozen w.r.t. DINO/graph math

        N, K, Hm, Wm = p_cnn.shape
        g = eigvecs.shape[1]
        ev_flat = eigvecs.reshape(N, g, Hm * Wm)                # [N, g, P]
        p_flat = p_cnn.reshape(N, K, Hm * Wm)                   # [N, K, P]

        # mu_k: [N, K, g] -- weighted average of eigvecs by P_CNN(.,k)
        weight_sum = p_flat.sum(dim=2, keepdim=True).clamp(min=1e-6)                  # [N, K, 1]
        mu = torch.bmm(p_flat, ev_flat.transpose(1, 2)) / weight_sum                  # [N, K, g]

        # squared distance from every pixel to every channel's own centroid
        ev_flat_t = ev_flat.transpose(1, 2)                                           # [N, P, g]
        dist2 = (ev_flat_t.unsqueeze(2) - mu.unsqueeze(1)).pow(2).sum(dim=3)          # [N, P, K]
        # Per-sample scale normalization: dist2's absolute magnitude depends
        # on grid_size/num_eigvecs/eigenvector normalization (measured on real
        # data, 260730: mean~0.0086 with grid_size=32/num_eigvecs=10 -- a raw
        # fixed temperature would need re-tuning any time those params change,
        # and wouldn't adapt to how spread-out/compact a given sample's own
        # clusters are). Dividing by the sample's own dist2 std makes
        # `temperature` a scale-free "how many std-devs of distance count as
        # meaningfully different" knob instead.
        dist2_scale = dist2.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        p_graph = F.softmax(-dist2 / dist2_scale / self.temperature, dim=2)           # [N, P, K]
        p_graph = p_graph.transpose(1, 2).reshape(N, K, Hm, Wm)                       # [N, K, Hm, Wm]

        if self.alpha != 1.0:
            p_graph = p_graph.clamp(min=1e-8).pow(self.alpha)

        fused = p_cnn * p_graph
        fused = fused / fused.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return fused

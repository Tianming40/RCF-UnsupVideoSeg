"""
DinoGraphFusionHead -- frozen-DINO graph-partitioning eigenvectors fused into
backbone feat0, discussed 260730 as a semantic-structure complement to v102's
motion-reconstruction mechanism ("前面的不丢,再加入这个提取到的信息 结合起来
一起做分割"). NOT a replacement for anything -- loss_warp_seg, decode_head,
decode_head2's architecture, and every existing loss are untouched; this is an
additive input-side module that sits between backbone2's feat0 and decode_head2.

Motivation / grounding (2408.14789v3, "Revisiting Surgical Instrument
Segmentation Without Human Intervention: A Graph Partitioning View"): a frozen
DINO ViT's patch features, turned into a cosine-similarity affinity graph and
spectrally decomposed (normalized graph Laplacian, Normalized-Cut theory), give
eigenvectors whose low-eigenvalue end carries clean object/module structure
(tool vs tissue vs background) -- this is a *label-free*, deterministic signal,
structurally different from this project's own ResNet50/DenseCL backbone
(verified 260729, saved/edge_test_260729/spectral_test.jpg: the identical
affinity/Laplacian/eigh pipeline run on this project's trained ResNet50
features produces pure noise, but produces clean instrument silhouettes on
DINO features -- so DINO, not ResNet, must be the feature source for this
mechanism).

Resolution / grid size (dino_input_size=384, grid_size=32) matches exactly the
configuration validated in that same spectral_test.jpg comparison: DINO patch8
on a 384x384 crop gives 48x48 patch tokens, bilinearly downsampled to a 32x32
(1024-node) graph before the affinity matrix -- not the smaller 128-input/16x16
configuration used elsewhere in this project for loss_dino, which has not been
separately validated for this purpose.

Everything through eigendecomposition runs under torch.no_grad() -- DINO is
frozen (shared with RCFDinoModel's own self.dino, used by loss_dino), and the
affinity/Laplacian/eigh pipeline is treated as a deterministic external signal,
exactly like RAFT flow or _extract_dino_feats elsewhere in this codebase. No
gradient flows into DINO or the graph math. The ONLY trainable component is a
small fusion conv module (eigenvector projection + concat + fuse-back conv),
whose final layer is zero-initialized so fused_feat0 == feat0 exactly at
init -- byte-identical to not having this module at all, so v102's own trained
optimum is preserved as the literal starting point. This mirrors why v144/v145
(image-edge boundary weight) didn't destabilize training the way the RAFT-free
JEPA line (v140-143) did: no new component starts in a cold, untrained state
that the rest of the network has to react to -- it starts as a no-op and only
gradually earns influence through gradient descent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoGraphFusionHead(nn.Module):
    def __init__(self, dino, dino_patch_size, feat_channels,
                 dino_input_size: int = 384, grid_size: int = 32,
                 num_eigvecs: int = 10, proj_channels: int = 32,
                 chunk_size: int = 8):
        super().__init__()
        self.dino = dino  # frozen _FrozenModule, shared with RCFDinoModel.dino
        self.dino_patch_size = dino_patch_size
        self.dino_input_size = dino_input_size
        self.grid_size = grid_size
        self.num_eigvecs = num_eigvecs
        # DINO-384's self-attention memory is O(N_tokens^2) per sample
        # (48x48=2304 tokens at patch8/384-input) -- chunk_size caps how many
        # frames go through DINO+eigh in one shot to bound peak memory. This
        # whole branch runs under no_grad (see _compute_eigvecs), so chunking
        # it has ZERO effect on training semantics -- batch_size and topk (the
        # easy-sample selection fraction, batch_size=8/topk=4 in v102) are
        # untouched. Do NOT "fix" a memory issue here by shrinking the actual
        # training batch_size -- that silently changes topk's survival ratio
        # (k = min(topk, batch_size)) and re-tunes an already-validated
        # mechanism as a side effect (discussed 260730).
        self.chunk_size = chunk_size

        self.eig_proj = nn.Conv2d(num_eigvecs, proj_channels, kernel_size=1)
        self.fuse_conv = nn.Conv2d(feat_channels + proj_channels, feat_channels, kernel_size=1)
        nn.init.zeros_(self.fuse_conv.weight)
        nn.init.zeros_(self.fuse_conv.bias)

    @torch.no_grad()
    def _compute_eigvecs(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, H, W], ImageNet-normalized (same convention as img_3 /
              backbone2's own input).
        Returns: [N, g, G, G] -- the g smallest-eigenvalue eigenvectors of the
                 normalized graph Laplacian (excluding the trivial constant
                 0th eigenvector), ascending eigenvalue order, per sample.

        Processes `imgs` in chunks of at most `self.chunk_size` frames to
        bound peak attention memory (see __init__ docstring) -- purely a
        memory-management detail, no_grad throughout, so chunk_size does not
        change the result (each frame's eigenvectors depend only on that
        frame, never on other frames in the same/different chunk) or affect
        training semantics in any way.
        """
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

    def forward(self, imgs: torch.Tensor, feat0: torch.Tensor) -> torch.Tensor:
        """
        imgs:  [N, 3, H, W]   -- same batch/order as feat0 (e.g. img_3, both
               frames of a training pair stacked into the batch dim).
        feat0: [N, C, H0, W0] -- backbone2's stage-0 feature map to fuse into.
        Returns: [N, C, H0, W0], == feat0 exactly at initialization.
        """
        eigvecs = self._compute_eigvecs(imgs)                                  # [N, g, G, G], no grad
        eigvecs = F.interpolate(eigvecs, size=feat0.shape[-2:], mode='bilinear', align_corners=False)
        eigvecs = eigvecs.detach()

        proj = self.eig_proj(eigvecs)
        fused = torch.cat([feat0, proj], dim=1)
        delta = self.fuse_conv(fused)
        return feat0 + delta

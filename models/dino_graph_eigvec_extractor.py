"""
DinoGraphEigvecExtractor: pure eigenvector extraction (frozen DINO -> affinity
-> Laplacian -> eigh), NO trainable parameters at all -- discussed 260801,
built for the "decoder-stage injection" line (models/multi_scale_seg_head_
decoder_prior.py's MultiScaleSegHeadDecoderPrior).

Unlike DinoGraphFusionHead (v146/v149) or DinoGraphAttentionGate (v153),
which each own BOTH the extraction logic AND a small trainable fusion
module bundled together, this class is deliberately extraction-ONLY. The
decoder-stage injection design keeps the trainable fusion module inside
decode_head2 itself (MultiScaleSegHeadDecoderPrior), analogous to how
models/multi_scale_seg_head_hires.py's stem-feature is captured externally
(a plain, parameter-free forward hook in RCFModel.__init__) and consumed by
a trainable module living inside decode_head2 -- keeping "frozen external
signal computation" and "trainable fusion" as two separate, independently
reasoned-about pieces rather than bundling them into one object living
outside decode_head2 (as v146/v149/v153 do, appropriate there since they
fuse INTO backbone2's own feature maps rather than into decode_head2's own
internal decoder-stage feature).

*** Isolation note: this file duplicates the same eigenvector-extraction
code used by DinoGraphFusionHead/DinoGraphEStepFusion/DinoGraphAttentionGate/
FlowAggregationHeadGraphMotion rather than sharing it, consistent with this
project's established practice of keeping each DINO-graph mechanism's code
path fully independent so concurrently-running/queued jobs are never at
risk from edits made for a different mechanism. ***
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoGraphEigvecExtractor(nn.Module):
    def __init__(self, dino, dino_patch_size,
                 dino_input_size: int = 384, grid_size: int = 32,
                 num_eigvecs: int = 10, chunk_size: int = 8):
        super().__init__()
        self.dino = dino  # frozen _FrozenModule, shared with RCFDinoModel.dino
        self.dino_patch_size = dino_patch_size
        self.dino_input_size = dino_input_size
        self.grid_size = grid_size
        self.num_eigvecs = num_eigvecs
        self.chunk_size = chunk_size

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: [N, 3, H, W] -> [N, g, G, G], no grad, frozen throughout."""
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

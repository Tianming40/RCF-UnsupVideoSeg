"""
DinoGraphAttentionGate: an alternative to DinoGraphFusionHead's concat-based
feature injection (models/dino_graph_fusion.py, used by v146/feat0 and
v149/feat3), discussed 260801 -- instead of concatenating a projection of
the DINO-graph eigenvectors directly into the backbone feature and letting a
3x3 conv learn to blend them, this predicts a single-channel SPATIAL GATE
from the eigenvectors and uses it to multiplicatively modulate the backbone
feature: "tell the network which regions to trust more," not "add new
content directly into the feature."

Motivation (260801, reviewing v149's channel-count imbalance): feat3 has
2048 channels, the eigenvector projection only 32 -- concatenated together,
eigenvector information is ~1.5% of the channels, and while v149's actual
trained weights (fuse_conv norm 0->10.17 after training) show the network
CAN learn to use this minority signal, a fusion mechanism that doesn't
depend on out-numbering the backbone feature in channel count is worth
testing on its own merits. Attention-gate fusion (Oktay et al.'s Attention
U-Net and many subsequent medical-segmentation papers) is a common
alternative: the auxiliary signal (here, the DINO graph structure) never
directly injects new content into the feature -- it only scales how much of
the EXISTING feature passes through at each spatial location.

F' = F * (1 + A), A = tanh(gate_logit), gate_logit computed from the
eigenvectors via a small conv stack whose LAST layer is zero-initialized --
so A = tanh(0) = 0 EXACTLY at initialization, and F' = F * (1 + 0) = F
EXACTLY, matching the same "exact, provable no-op at init" property used by
every other DINO-graph mechanism this session (v146/v149's zero-init residual
delta, v147's alpha=0 special case). A is bounded in [-1, 1], so F' ranges
over F * [0, 2] once trained -- can suppress a location toward 0 or boost it
up to 2x, both smoothly, never a hard binary decision.

*** Isolation note: this file duplicates DinoGraphFusionHead's eigenvector-
extraction code (DINO forward / affinity / Laplacian / eigh) rather than
sharing it, and is wired via a NEW, separate toggle/attribute/hook -- v149's
own dino_graph_fusion_deep_head code path (and the running v151/v152 jobs
that depend on models/rcf_model.py and models/flow_aggregation_head_with_
residual_v2.py, which this file does not touch) is completely untouched. ***
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoGraphAttentionGate(nn.Module):
    def __init__(self, dino, dino_patch_size, feat_channels,
                 dino_input_size: int = 384, grid_size: int = 32,
                 num_eigvecs: int = 10, gate_hidden_channels: int = 32,
                 chunk_size: int = 8):
        super().__init__()
        self.dino = dino  # frozen _FrozenModule, shared with RCFDinoModel.dino
        self.dino_patch_size = dino_patch_size
        self.dino_input_size = dino_input_size
        self.grid_size = grid_size
        self.num_eigvecs = num_eigvecs
        self.chunk_size = chunk_size

        # feat_channels kept as a constructor arg for interface parity with
        # DinoGraphFusionHead (and future configs that might want a per-
        # channel gate instead of a single shared spatial map) -- unused by
        # the current single-channel-gate design, which broadcasts across
        # all of feat_channels uniformly.
        self.feat_channels = feat_channels

        self.gate_conv1 = nn.Sequential(
            nn.Conv2d(num_eigvecs, gate_hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate_conv2 = nn.Conv2d(gate_hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.gate_conv2.weight)
        nn.init.zeros_(self.gate_conv2.bias)

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

    def forward(self, imgs: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]     -- same batch/order as feat (e.g. img_3).
        feat: [N, C, H', W']   -- backbone2 feature map to gate.
        Returns: [N, C, H', W'], == feat exactly at initialization.
        """
        eigvecs = self._compute_eigvecs(imgs)                                  # [N, g, G, G], no grad, frozen
        eigvecs = F.interpolate(eigvecs, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        eigvecs = eigvecs.detach()

        h = self.gate_conv1(eigvecs)
        logit = self.gate_conv2(h)             # [N, 1, H', W'], zero-init -> 0 at init
        gate = torch.tanh(logit)               # [N, 1, H', W'], in [-1, 1], == 0 at init
        return feat * (1.0 + gate)             # broadcasts over channel dim; == feat exactly at init

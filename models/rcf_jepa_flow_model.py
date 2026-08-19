"""
RCFJepaFlowModel: JEPA-style (Assran et al. 2023, I-JEPA; Bardes et al. 2024,
V-JEPA) asymmetric target for the self-taught flow branch, discussed 260728
after diagnosing WHY RCFSelfTaughtFlowModel (v140) kept sliding toward
degenerate/collapsed solutions even after the zero-flow-collapse fixes
(baseline-relative + weighted loss_flow_recon) and a k-means-informed mask
init: the reconstruction target (the OTHER frame's features) was produced by
the SAME, jointly-trained backbone computing the prediction -- the model was
effectively grading its own homework. Combined with the aperture problem
(many different flow fields give equally-low reconstruction error in
low-texture regions -- see session discussion), this self-referential setup
has too much freedom and settles into internally-consistent-but-physically-
meaningless solutions (mask collapsing to one channel, flow collapsing near
zero) whenever the ambiguous, texture-poor majority of the frame dominates
the (unweighted or weighted) average loss.

JEPA's fix (used by BYOL/DINO too, same underlying principle): predict a
TARGET encoder's output, where the target encoder is a stop-gradient,
EMA-updated shadow copy of the trained ("context") encoder, NOT the same
network computing both sides. This breaks the model-grades-itself symmetry
that self-referential reconstruction/siamese setups are prone to collapsing
into, without needing an externally pretrained "answer key" (which doesn't
exist for flow -- that's exactly why RAFT was removed in the first place).

Architecture change vs RCFSelfTaughtFlowModel (v140), everything else
UNCHANGED:
  - backbone2 (context encoder, actively trained) processes BOTH frames, as
    always -- feeds the mask branch (joint_attn0..3 -> decode_head2,
    completely untouched) and the QUERY side of flow_head's correlation.
  - backbone2_ema (target encoder, EMA-updated, requires_grad=False --
    REUSES this project's existing EMA infrastructure, models/rcf_model.py's
    create_backbone_with_ema/momentum_update_param_and_buffer, previously
    unused by any RAFT-free config) processes BOTH frames too, feeds ONLY
    the KEY/VALUE side of flow_head's correlation (models/
    local_correlation_flow_head.py:forward_asymmetric, new method, reuses
    the SAME correlation math as v140 -- only which encoder's output is
    routed into query vs key/value differs).
  - loss_flow_recon's target (the "other frame" feature being warped
    towards) now comes from backbone2_ema instead of backbone2 -- same
    baseline-relative, baseline-weighted formulation as v140 (see that
    file's forward_train comments for the full rationale on both), just
    computed against an asymmetric target. baseline_err_fw and
    baseline_err_bw are no longer identical (v140 could reuse one value for
    both directions since both sides came from the same encoder --
    context-vs-EMA is no longer symmetric that way, computed separately).
  - decode_head (FlowAggregationHeadRaftFree), topk, mask branch, k-means
    conv_seg init (kmeans_init_mask_head, inherited unchanged -- operates
    entirely on backbone2/joint_attn/decode_head2, untouched by this
    change), smoothness/cycle losses, DINO loss: all UNCHANGED, inherited
    from RCFSelfTaughtFlowModel without modification.

Config must set model_kwargs.backbone2.create_ema: true (and optionally
ema_m, default 0.999 per RCFModel) for backbone2_ema to actually be built --
without it this class would crash (self.backbone2_ema is None).
"""
import torch
import torch.nn.functional as F

import utils
from models.rcf_selftaught_flow_model import RCFSelfTaughtFlowModel
from utils.warp_utils import flow_warp, edge_aware_smoothness_loss

logger = utils.get_logger()


class RCFJepaFlowModel(RCFSelfTaughtFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.backbone2_ema is not None, (
            "RCFJepaFlowModel requires an EMA target encoder -- set "
            "model_kwargs.backbone2.create_ema: true in the config.")

    def forward_train(self, imgs, seq_ids, seq_names, paths, gaps=None):
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        assert im_num == 2, "RCFJepaFlowModel requires paired frames (im_num=2)"

        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)          # context (trainable)
        all_feat_ema = self.extract_feat(img_3, self.backbone2_ema)  # target (EMA, no grad)

        # mask branch -- UNCHANGED, still purely on context-encoder features
        # (no self-referential-target problem here, only flow's reconstruction
        # loss had one -- see module docstring).
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        if self.allow_mask_resize and all_pred_mask.shape[-2:] != self.mask_size:
            all_pred_mask = self.resize(all_pred_mask, self.mask_size)

        if self.separate_residual:
            all_pred_residual_fw, all_pred_residual_bw = self.pred_separate_residual(all_feat, batch_size, im_num)
        else:
            all_pred_residual_fw, all_pred_residual_bw = self.pred_joint_residual(
                all_feat[-1].unflatten(0, (batch_size, im_num)))

        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        # ── JEPA-style self-taught flow: query=context, key/value=EMA target ──
        flow_raw_fw, flow_raw_bw = self.flow_head.forward_asymmetric(
            all_feat[2], all_feat_ema[2], all_feat[0], all_feat_ema[0])

        feat0_hw = all_feat[0].shape[-2:]
        feat0 = all_feat[0].view(batch_size, im_num, *all_feat[0].shape[1:])
        feat0_i, feat0_j = feat0[:, 0], feat0[:, 1]                          # context, both frames
        feat0_ema = all_feat_ema[0].view(batch_size, im_num, *all_feat_ema[0].shape[1:])
        feat0_i_ema, feat0_j_ema = feat0_ema[:, 0], feat0_ema[:, 1]          # EMA target, both frames

        # Predict the TARGET encoder's features at the matched location, not
        # the context encoder's own (self-referential) features -- the core
        # JEPA substitution, see module docstring.
        recon_err_fw = (feat0_i - flow_warp(feat0_j_ema, flow_raw_fw, pad='border')).abs().mean(dim=1, keepdim=True)
        recon_err_bw = (feat0_j - flow_warp(feat0_i_ema, flow_raw_bw, pad='border')).abs().mean(dim=1, keepdim=True)

        # Baseline-relative + baseline-weighted, same rationale as
        # RCFSelfTaughtFlowModel (v140) -- see that file's forward_train for
        # the full writeup on both. Only difference: fw/bw baselines are no
        # longer identical (context and EMA-target encoders differ, even
        # though EMA started as a copy -- it lags behind), so both are
        # computed explicitly rather than reusing one value for both
        # directions like v140 could.
        baseline_err_fw = (feat0_i.detach() - feat0_j_ema).abs().mean(dim=1, keepdim=True)
        baseline_err_bw = (feat0_j.detach() - feat0_i_ema).abs().mean(dim=1, keepdim=True)

        eps = 1e-6
        loss_flow_recon_fw = ((recon_err_fw - baseline_err_fw) * baseline_err_fw).sum(dim=(1, 2, 3)) / (baseline_err_fw.sum(dim=(1, 2, 3)) + eps)
        loss_flow_recon_bw = ((recon_err_bw - baseline_err_bw) * baseline_err_bw).sum(dim=(1, 2, 3)) / (baseline_err_bw.sum(dim=(1, 2, 3)) + eps)
        loss_flow_recon = loss_flow_recon_fw.mean() + loss_flow_recon_bw.mean()

        img0_small = self.resize(imgs[:, 0], feat0_hw)
        img1_small = self.resize(imgs[:, 1], feat0_hw)
        loss_flow_smooth = (edge_aware_smoothness_loss(flow_raw_fw, img0_small)
                             + edge_aware_smoothness_loss(flow_raw_bw, img1_small))

        cycle_err = (flow_raw_fw + flow_warp(flow_raw_bw, flow_raw_fw, pad='border')).abs().mean(dim=1, keepdim=True)
        loss_flow_cycle = cycle_err.mean()

        loss_flow_selftaught = (loss_flow_recon
                                 + self.w_flow_smooth * loss_flow_smooth
                                 + self.w_flow_cycle * loss_flow_cycle)

        conf_fw = torch.exp(-recon_err_fw.detach() / self.recon_conf_sigma)
        conf_bw = torch.exp(-recon_err_bw.detach() / self.recon_conf_sigma)
        self.decode_head.set_external_weights(conf_fw, conf_bw)

        downsample_factor = _w / feat0_hw[-1]
        flow_raw_fw_in = (flow_raw_fw * downsample_factor).detach().unsqueeze(1)
        flow_raw_bw_in = (flow_raw_bw * downsample_factor).detach().unsqueeze(1)

        pred_flows, loss_flow = self.decode_head(
            imgs, all_pred_mask, flow_raw_fw_in, flow_raw_bw_in,
            all_pred_residual_fw, all_pred_residual_bw, seq_names=seq_names, gaps=gaps)

        loss_warp_seg = loss_flow['seg']

        if self.train_iter % self.log_interval == 0:
            self._save_train_viz(all_pred_mask, imgs, img0_small, img1_small,
                                  flow_raw_fw, flow_raw_bw, recon_err_fw, recon_err_bw,
                                  conf_fw, conf_bw, batch_size, im_num,
                                  seq_names, seq_ids, paths)

        losses = {
            "loss_warp_seg": loss_warp_seg,
            "loss_flow_recon": loss_flow_recon,
            "loss_flow_smooth": loss_flow_smooth,
            "loss_flow_cycle": loss_flow_cycle,
        }
        loss = loss_warp_seg * self.w_seg + loss_flow_selftaught * self.w_flow_recon

        if self.w_dino > 0.0:
            l_dino = self._dino_consistency_loss(all_pred_mask[:, 0], imgs[:, 0])
            losses["loss_dino"] = l_dino
            loss = loss + self.w_dino * l_dino

        losses["loss"] = loss

        # EMA momentum update -- backbone2_ema NEVER receives gradient
        # directly (utils.set_no_grad in create_backbone_with_ema); this is
        # its only update mechanism, a slow exponential shadow of backbone2.
        utils.momentum_update_param_and_buffer(src=self.backbone2, dest=self.backbone2_ema, m=self.ema_m)

        self.train_iter += 1
        return losses

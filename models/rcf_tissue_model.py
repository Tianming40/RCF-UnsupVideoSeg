"""
rcf_tissue_model.py — RCFTissueModel (V2)

V2 changes vs V1:
  - L_compact removed  (caused instrument masks to fragment)
  - L_rigid added      (penalises intra-channel flow variance — physically correct)
  - L_grasp_conv added (flow-divergence prior guides grasping-point channel ch2)
  - L_deform added     (tissue residual R̂ > background residual)
  - w_align  0.10 → 0.05
  - w_motion 0.05 → 0.03
  - Captures residual_fw via pred_separate_residual override (for L_deform)

Design: subclasses RCFModel, does NOT modify any existing file.
"""

import torch
import torch.nn.functional as F

from .rcf_model import RCFModel
from .tissue_role_loss import (
    rigidity_loss,
    grasping_convergence_loss,
    deformation_loss,
    tissue_flow_alignment_loss,
    tissue_motion_loss,
)

import utils
logger = utils.get_logger()


class RCFTissueModel(RCFModel):
    """
    RCFModel + tissue-role auxiliary losses (V2).

    Extra __init__ parameters (all keyword-only with defaults):
      instrument_channels  list[int]   channels that receive L_rigid
      tissue_channel       int         soft-tissue channel (ch1)
      grasping_channel     int         instrument-tip channel (ch2, highest convergence)
      bg_channels          list[int]   background channels
      w_rigid              float       weight for L_rigid
      w_grasp_conv         float       weight for L_grasp_conv (convergence prior)
      w_deform             float       weight for L_deform
      w_align              float       weight for L_align
      w_motion             float       weight for L_motion
      motion_margin        float       hinge margin (px) for L_motion
      deform_margin        float       hinge margin for L_deform
      min_grasp_frac       float       suppress L_align when ch_grasp area < frac×H×W
    """

    def __init__(
            self,
            args,
            instrument_channels=(0, 2, 3),
            tissue_channel: int = 1,
            grasping_channel: int = 2,
            bg_channels=(4,),
            w_rigid: float = 0.0,
            w_grasp_conv: float = 0.0,
            w_deform: float = 0.0,
            w_align: float = 0.0,
            w_motion: float = 0.0,
            motion_margin: float = 1.0,
            deform_margin: float = 0.5,
            min_grasp_frac: float = 0.005,
            # kept for backward-compat with V1 configs (ignored)
            w_compact: float = 0.0,
            **kwargs,
    ):
        super().__init__(args, **kwargs)

        self.instrument_channels = list(instrument_channels)
        self.tissue_channel      = tissue_channel
        self.grasping_channel    = grasping_channel
        self.bg_channels         = tuple(bg_channels)
        self.w_rigid             = w_rigid
        self.w_grasp_conv        = w_grasp_conv
        self.w_deform            = w_deform
        self.w_align             = w_align
        self.w_motion            = w_motion
        self.motion_margin       = motion_margin
        self.deform_margin       = deform_margin
        self.min_grasp_frac      = min_grasp_frac

        self._captured_mask_logits  = None
        self._captured_residual_fw  = None

        logger.info(
            "[RCFTissueModel V2] inst=%s  tissue=%d  grasp=%d  bg=%s  "
            "w_rigid=%.3f  w_grasp_conv=%.3f  w_deform=%.3f  "
            "w_align=%.3f  w_motion=%.3f",
            self.instrument_channels, self.tissue_channel,
            self.grasping_channel, list(self.bg_channels),
            w_rigid, w_grasp_conv, w_deform, w_align, w_motion,
        )

    # ── capture mask logits (override instead of hook — hook bypassed by .forward()) ──

    def _decode_head_forward(self, x, decode_head):
        pred = decode_head.forward(x)
        if self.training and decode_head is self.decode_head2:
            self._captured_mask_logits = pred          # [B*I, C, fH, fW]
        return pred

    # ── capture residual (needed for L_deform) ────────────────────────────────

    def pred_separate_residual(self, feats, batch_size, im_num):
        res_fw, res_bw = super().pred_separate_residual(feats, batch_size, im_num)
        if self.training:
            self._captured_residual_fw = res_fw        # [B, 2*C, fH, fW]
        return res_fw, res_bw

    # ── forward_train override ────────────────────────────────────────────────

    def forward_train(self, imgs, seq_ids, seq_names, paths,
                      gt_fw_flows, gt_bw_flows, pl_masks):

        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths,
            gt_fw_flows, gt_bw_flows, pl_masks,
        )

        if self._captured_mask_logits is None:
            return losses

        # ── reconstruct soft masks at mask_size ───────────────────────────────
        B, I = imgs.shape[0], imgs.shape[1]
        raw = self._captured_mask_logits
        self._captured_mask_logits = None

        if self.allow_mask_resize and raw.shape[-2:] != torch.Size(list(self.mask_size)):
            raw = self.resize(raw, self.mask_size)
        raw   = raw.view(B, I, self.mask_layer, *self.mask_size)
        masks = F.softmax(raw, dim=2)                  # [B, I, C, H, W]
        masks0 = masks[:, 0]                           # [B, C, H, W]  frame-0

        # ── forward flow at mask resolution ───────────────────────────────────
        flow = gt_fw_flows[:, 0]
        if flow.ndim == 4 and flow.shape[-1] == 2:
            flow = flow.permute(0, 3, 1, 2).contiguous()
        flow_r   = self.resize(flow, self.mask_size)   # [B, 2, H, W]
        flow_mag = flow_r.norm(dim=1)                  # [B, H, W]

        # ── 1. Rigidity loss — instrument channels ────────────────────────────
        if self.w_rigid > 0:
            L_rigid = rigidity_loss(masks0, flow_r, self.instrument_channels)
            losses['loss_rigid'] = L_rigid
            losses['loss'] = losses['loss'] + self.w_rigid * L_rigid

        # ── 2. Grasping-point convergence loss (convergence prior) ────────────
        if self.w_grasp_conv > 0:
            L_grasp = grasping_convergence_loss(masks0, flow_r, self.grasping_channel)
            losses['loss_grasp_conv'] = L_grasp
            losses['loss'] = losses['loss'] + self.w_grasp_conv * L_grasp

        # ── 3. Deformation loss — tissue residual > background ────────────────
        if self.w_deform > 0 and self._captured_residual_fw is not None:
            res_fw = self._captured_residual_fw
            self._captured_residual_fw = None
            if res_fw.shape[-2:] != torch.Size(list(self.mask_size)):
                res_fw = self.resize(res_fw, self.mask_size)
            L_deform = deformation_loss(
                masks0, res_fw,
                tissue_channel=self.tissue_channel,
                bg_channels=self.bg_channels,
                n_classes=self.num_classes,
                pred_div_coeff=self.decode_head.pred_div_coeff,
                residual_scale=self.decode_head.residual_adjustment_scale,
                margin=self.deform_margin,
            )
            losses['loss_deform'] = L_deform
            losses['loss'] = losses['loss'] + self.w_deform * L_deform
        else:
            self._captured_residual_fw = None          # clear even if unused

        # ── 4. Tissue–grasping flow alignment ─────────────────────────────────
        if self.w_align > 0:
            L_align = tissue_flow_alignment_loss(
                masks0, flow_r,
                grasping_channel=self.grasping_channel,
                tissue_channel=self.tissue_channel,
                min_grasp_frac=self.min_grasp_frac,
            )
            losses['loss_align'] = L_align
            losses['loss'] = losses['loss'] + self.w_align * L_align

        # ── 5. Tissue motion > background motion ──────────────────────────────
        if self.w_motion > 0:
            L_motion = tissue_motion_loss(
                masks0, flow_mag,
                tissue_channel=self.tissue_channel,
                bg_channels=self.bg_channels,
                margin=self.motion_margin,
            )
            losses['loss_motion'] = L_motion
            losses['loss'] = losses['loss'] + self.w_motion * L_motion

        return losses

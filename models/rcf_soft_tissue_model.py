"""
rcf_soft_tissue_model.py — RCFSoftTissueModel

Inherits RCFDinoModel and combines ALL tissue-role losses from
rcf_tissue_model.py with the annotation-driven losses and KL distillation
added for grasp0.  rcf_tissue_model.py is superseded by this file.

Loss set (all weights default to 0.0 — enable in config):

  Inherited from RCFDinoModel:
    L_dino         : DINO visual-consistency (w_dino)

  From RCFTissueModel V2 (unsupervised):
    L_rigid        : instrument channels have uniform internal flow (w_rigid)
    L_grasp_conv   : grasping channel aligns with flow convergence (w_grasp_conv)
    L_deform       : tissue residual > background residual (w_deform)
    L_align        : tissue flow aligns with grasping-channel mean flow (w_align)
    L_motion       : tissue motion > background motion (w_motion)

  Annotation-driven V3 (grasp0-specific):
    L_grasp_flow   : tissue activates where flow matches grasp-point flow,
                     excluding existing instrument region (w_grasp_flow)
    L_dissect_prox : tissue activates near annotated dissection point (w_dissect)

  KL distillation:
    L_distill      : BCE(student_ch, teacher_ch) — preserves pretrained
                     instrument-detector channel from degrading (w_distill)

Usage:
    python main_grasp0.py  configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rcf_dino_model import RCFDinoModel, _FrozenModule
from .rcf_model import RCFModel
from .tissue_role_loss import (
    rigidity_loss,
    grasping_convergence_loss,
    deformation_loss,
    tissue_flow_alignment_loss,
    tissue_motion_loss,
    grasp_flow_alignment_loss,
    dissect_proximity_loss,
)

import utils
logger = utils.get_logger()

_DINO_ONLY_KWARGS = frozenset({
    'dino_arch', 'dino_patch_size', 'dino_checkpoint', 'dino_input_size', 'w_dino',
})


class RCFSoftTissueModel(RCFDinoModel):
    """
    RCFDinoModel + full tissue-role loss set + KL distillation.

    All tissue/annotation kwargs are keyword-only with safe defaults.
    Pass them under  model_kwargs:  in the YAML config.

    Channel convention (grasp0 fine-tune):
      instrument_channel  = 1   (pretrained grasp10 instrument, protected by L_distill)
      instrument_channels = [1] (for L_rigid)
      tissue_channel      = 2   (new soft-tissue channel to train)
      grasping_channel    = 1   (same as instrument for L_align / L_grasp_conv)
      bg_channels         = [0, 3, 4]
      distill_channel     = 1   (same as instrument_channel)
    """

    def __init__(
        self,
        args,
        # ── channel assignments ──────────────────────────────────────────
        instrument_channels=(1,),
        instrument_channel: int = 1,
        tissue_channel: int = 2,
        grasping_channel: int = 1,
        bg_channels=(0, 3, 4),
        # ── V2 unsupervised losses ────────────────────────────────────────
        w_rigid: float = 0.0,
        w_grasp_conv: float = 0.0,
        w_deform: float = 0.0,
        w_align: float = 0.0,
        w_motion: float = 0.0,
        motion_margin: float = 1.0,
        deform_margin: float = 0.5,
        min_grasp_frac: float = 0.005,
        # ── V3 annotation-driven losses ───────────────────────────────────
        w_grasp_flow: float = 0.0,
        w_dissect: float = 0.0,
        grasp_flow_min_mag: float = 0.5,
        dissect_sigma: float = 0.12,
        # ── KL distillation ───────────────────────────────────────────────
        w_distill: float = 0.0,
        teacher_ckpt: Optional[str] = None,
        distill_channel: int = 1,
        # ── backward compat with old configs ─────────────────────────────
        w_compact: float = 0.0,
        **kwargs,
    ):
        super().__init__(args, **kwargs)

        self.instrument_channels = list(instrument_channels)
        self.instrument_channel  = instrument_channel
        self.tissue_channel      = tissue_channel
        self.grasping_channel    = grasping_channel
        self.bg_channels         = tuple(bg_channels)

        self.w_rigid         = w_rigid
        self.w_grasp_conv    = w_grasp_conv
        self.w_deform        = w_deform
        self.w_align         = w_align
        self.w_motion        = w_motion
        self.motion_margin   = motion_margin
        self.deform_margin   = deform_margin
        self.min_grasp_frac  = min_grasp_frac

        self.w_grasp_flow       = w_grasp_flow
        self.w_dissect          = w_dissect
        self.grasp_flow_min_mag = grasp_flow_min_mag
        self.dissect_sigma      = dissect_sigma

        self.w_distill       = w_distill
        self.distill_channel = distill_channel

        # per-batch state (set in forward, consumed in forward_train)
        self._batch_grasp_xy      = None
        self._batch_dissect_xy    = None
        self._captured_residual_fw = None

        # frozen teacher for KL distillation
        self._teacher = None
        if w_distill > 0 and teacher_ckpt is not None:
            self._teacher = self._build_frozen_teacher(teacher_ckpt, kwargs)
            logger.info("[RCFSoftTissueModel] Distill teacher: %s", teacher_ckpt)

        logger.info(
            "[RCFSoftTissueModel] "
            "inst=%s  inst_ch=%d  tissue=%d  grasp=%d  bg=%s\n"
            "  w_rigid=%.3f  w_grasp_conv=%.3f  w_deform=%.3f  "
            "w_align=%.3f  w_motion=%.3f\n"
            "  w_grasp_flow=%.3f  w_dissect=%.3f  w_distill=%.3f",
            self.instrument_channels, instrument_channel,
            tissue_channel, grasping_channel, list(self.bg_channels),
            w_rigid, w_grasp_conv, w_deform, w_align, w_motion,
            w_grasp_flow, w_dissect, w_distill,
        )

    # ── Frozen teacher ─────────────────────────────────────────────────────────

    def _build_frozen_teacher(self, ckpt_path: str, base_kwargs: dict) -> nn.Module:
        """
        Lightweight teacher: RCFModel (backbone2 + decode_head2, no DINO).
        Loaded from a PL checkpoint with strict=False.
        Wrapped in _FrozenModule so it stays in eval mode permanently.
        """
        rcf_kwargs = {k: v for k, v in base_kwargs.items()
                      if k not in _DINO_ONLY_KWARGS}
        teacher = RCFModel(args=self.args, **rcf_kwargs)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd   = ckpt.get('state_dict', ckpt)
        if any(k.startswith('model.') for k in sd):
            sd = {k[len('model.'):]: v for k, v in sd.items()
                  if k.startswith('model.')}
        m = teacher.load_state_dict(sd, strict=False)
        logger.info("[Teacher] missing=%d  unexpected=%d",
                    len(m.missing_keys), len(m.unexpected_keys))
        return _FrozenModule(teacher)

    # ── Capture annotation tensors ─────────────────────────────────────────────

    def forward(self, x, return_pred_vis_list=False):
        if self.training:
            self._batch_grasp_xy   = x.get('grasp_xy',   None)
            self._batch_dissect_xy = x.get('dissect_xy', None)
        return super().forward(x, return_pred_vis_list=return_pred_vis_list)

    # ── Capture residual (needed for L_deform) ─────────────────────────────────

    def pred_separate_residual(self, feats, batch_size, im_num):
        res_fw, res_bw = super().pred_separate_residual(feats, batch_size, im_num)
        if self.training:
            self._captured_residual_fw = res_fw   # [B, 2*C, fH, fW]
        return res_fw, res_bw

    # ── Teacher ch-prob ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _teacher_ch_prob(self, imgs_b3hw: torch.Tensor) -> torch.Tensor:
        """backbone2 + decode_head2 of frozen teacher → softmax [B, C, Hm, Wm]."""
        tm    = self._teacher.module
        feats = tm.extract_feat(imgs_b3hw, tm.backbone2)
        logits = tm._decode_head_forward(feats, tm.decode_head2)
        if logits.shape[-2:] != torch.Size(list(self.mask_size)):
            logits = self.resize(logits, self.mask_size)
        return F.softmax(logits, dim=1)

    # ── forward_train ──────────────────────────────────────────────────────────

    def forward_train(self, imgs, seq_ids, seq_names, paths,
                      gt_fw_flows, gt_bw_flows, pl_masks):
        # super() = RCFDinoModel which:
        #   1. resets _captured_mask_logits
        #   2. runs RCFModel.forward_train (sets _captured_mask_logits, calls
        #      pred_separate_residual which sets _captured_residual_fw)
        #   3. appends L_dino
        #   4. returns WITHOUT clearing _captured_mask_logits
        losses = super().forward_train(
            imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks,
        )

        if self._captured_mask_logits is None:
            return losses

        # ── reconstruct soft masks ─────────────────────────────────────────────
        B, I = imgs.shape[0], imgs.shape[1]
        raw = self._captured_mask_logits        # [B*I, C, fH, fW]
        self._captured_mask_logits = None

        C = raw.shape[1]
        if self.allow_mask_resize and raw.shape[-2:] != torch.Size(list(self.mask_size)):
            raw = self.resize(raw, self.mask_size)
        raw    = raw.view(B, I, C, *raw.shape[-2:])
        masks  = F.softmax(raw, dim=2)          # [B, I, C, Hm, Wm]
        masks0 = masks[:, 0]                    # [B, C, Hm, Wm]  frame-0

        # ── forward flow at mask resolution ───────────────────────────────────
        flow = gt_fw_flows[:, 0]
        if flow.ndim == 4 and flow.shape[-1] == 2:
            flow = flow.permute(0, 3, 1, 2).contiguous()
        flow_r   = self.resize(flow, self.mask_size)   # [B, 2, Hm, Wm]
        flow_mag = flow_r.norm(dim=1)                  # [B, Hm, Wm]

        # ── annotation tensors ────────────────────────────────────────────────
        grasp_xy   = self._batch_grasp_xy
        dissect_xy = self._batch_dissect_xy
        self._batch_grasp_xy   = None
        self._batch_dissect_xy = None

        # ── 1. L_rigid ────────────────────────────────────────────────────────
        if self.w_rigid > 0:
            L = rigidity_loss(masks0, flow_r, self.instrument_channels)
            losses['loss_rigid'] = L
            losses['loss'] = losses['loss'] + self.w_rigid * L

        # ── 2. L_grasp_conv ───────────────────────────────────────────────────
        if self.w_grasp_conv > 0:
            L = grasping_convergence_loss(masks0, flow_r, self.grasping_channel)
            losses['loss_grasp_conv'] = L
            losses['loss'] = losses['loss'] + self.w_grasp_conv * L

        # ── 3. L_deform ───────────────────────────────────────────────────────
        res_fw = self._captured_residual_fw
        self._captured_residual_fw = None
        if self.w_deform > 0 and res_fw is not None:
            if res_fw.shape[-2:] != torch.Size(list(self.mask_size)):
                res_fw = self.resize(res_fw, self.mask_size)
            L = deformation_loss(
                masks0, res_fw,
                tissue_channel=self.tissue_channel,
                bg_channels=self.bg_channels,
                n_classes=self.num_classes,
                pred_div_coeff=self.decode_head.pred_div_coeff,
                residual_scale=self.decode_head.residual_adjustment_scale,
                margin=self.deform_margin,
            )
            losses['loss_deform'] = L
            losses['loss'] = losses['loss'] + self.w_deform * L

        # ── 4. L_align ────────────────────────────────────────────────────────
        if self.w_align > 0:
            L = tissue_flow_alignment_loss(
                masks0, flow_r,
                grasping_channel=self.grasping_channel,
                tissue_channel=self.tissue_channel,
                min_grasp_frac=self.min_grasp_frac,
            )
            losses['loss_align'] = L
            losses['loss'] = losses['loss'] + self.w_align * L

        # ── 5. L_motion ───────────────────────────────────────────────────────
        if self.w_motion > 0:
            L = tissue_motion_loss(
                masks0, flow_mag,
                tissue_channel=self.tissue_channel,
                bg_channels=self.bg_channels,
                margin=self.motion_margin,
            )
            losses['loss_motion'] = L
            losses['loss'] = losses['loss'] + self.w_motion * L

        # ── 6. L_grasp_flow ───────────────────────────────────────────────────
        if self.w_grasp_flow > 0 and grasp_xy is not None:
            L = grasp_flow_alignment_loss(
                masks0, flow_r,
                tissue_channel=self.tissue_channel,
                instrument_channel=self.instrument_channel,
                grasp_xy=grasp_xy,
                min_flow_mag=self.grasp_flow_min_mag,
            )
            losses['loss_grasp_flow'] = L
            losses['loss'] = losses['loss'] + self.w_grasp_flow * L

        # ── 7. L_dissect_prox ─────────────────────────────────────────────────
        if self.w_dissect > 0 and dissect_xy is not None:
            L = dissect_proximity_loss(
                masks0,
                tissue_channel=self.tissue_channel,
                dissect_xy=dissect_xy,
                sigma=self.dissect_sigma,
            )
            losses['loss_dissect'] = L
            losses['loss'] = losses['loss'] + self.w_dissect * L

        # ── 8. L_distill ──────────────────────────────────────────────────────
        if self.w_distill > 0 and self._teacher is not None:
            with torch.no_grad():
                teacher_p = self._teacher_ch_prob(imgs[:, 0])
            tc        = self.distill_channel
            student_p = masks0[:, tc].clamp(1e-6, 1 - 1e-6)
            target_p  = teacher_p[:, tc].detach()
            L = F.binary_cross_entropy(student_p, target_p, reduction='mean')
            losses['loss_distill'] = L
            losses['loss'] = losses['loss'] + self.w_distill * L

        return losses

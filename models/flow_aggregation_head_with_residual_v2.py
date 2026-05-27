"""
FlowAggregationHeadWithResidualV2

Extends FlowAggregationHeadWithResidual with two tunable parameters:

  topk              (int,   default=4)         – number of easiest samples
                                                 selected per batch (was hard-
                                                 coded to 2 in V1)

  boundary_threshold (float, default=pi/18)    – angle-change threshold for
                                                 detect_flow_changes_batch
                                                 (was hard-coded to pi/12 in
                                                 V1; smaller = more sensitive,
                                                 better for CMC small-motion)

Usage in config YAML:
  decode_head:
    type: FlowAggregationHeadWithResidualV2
    topk: 4
    boundary_threshold: 0.1745   # pi/18 ≈ 0.1745
    ...  (all other kwargs same as V1)
"""

import math
import torch
import torch.nn.functional as F

from .flow_aggregation_head_with_residual import FlowAggregationHeadWithResidual


class FlowAggregationHeadWithResidualV2(FlowAggregationHeadWithResidual):
    """
    V2: topk and boundary_threshold are configurable via __init__.
    Everything else is identical to FlowAggregationHeadWithResidual.
    """

    def __init__(self, *args, topk: int = 4,
                 boundary_threshold: float = math.pi / 18,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert topk >= 1, "topk must be >= 1"
        self.topk = topk
        self.boundary_threshold = boundary_threshold

    # ------------------------------------------------------------------ #
    # Override detect_flow_changes_batch to use self.boundary_threshold   #
    # ------------------------------------------------------------------ #
    def detect_flow_changes_batch(self, flow_data,
                                  threshold=None,
                                  dilation_size=7):
        """Same as V1 but uses self.boundary_threshold when threshold=None."""
        if threshold is None:
            threshold = self.boundary_threshold
        return super().detect_flow_changes_batch(
            flow_data, threshold=threshold, dilation_size=dilation_size)

    # ------------------------------------------------------------------ #
    # Override forward to use self.topk instead of hard-coded 2           #
    # ------------------------------------------------------------------ #
    def forward(self, imgs, masks, gt_fw_flows, gt_bw_flows,
                all_pred_residual_fw, all_pred_residual_bw):

        flow_loss = {'seg_fw': 0., 'seg_bw': 0.}
        flows = {'gt_flow': [], 'pred_flow': [], 'agg_flow': [],
                 'residual_adj': [], 'affine_flow': []}

        batch_size, im_num, _, im_h, im_w = imgs.shape
        assert im_num == 2, "Other im_num not implemented"

        individual_losses_fw = []
        individual_losses_bw = []

        from .flow_aggregation_head_with_residual import get_norm_flow

        for i in range(1, im_num):
            mask1 = masks[:, i - 1, :, :, :]
            mask2 = masks[:, i, :, :, :]

            gt_fw_flow = gt_fw_flows[:, i - 1, ...]
            gt_bw_flow = gt_bw_flows[:, i - 1, ...]

            gt_fw_flow = self.norm_and_clamp_flow(gt_fw_flow)
            gt_bw_flow = self.norm_and_clamp_flow(gt_bw_flow)

            fw_flow_overall, fw_flow_agg, fw_residual_adjustment, fw_flow_affine = \
                self.aggregate_flow_with_residual(mask1, gt_fw_flow, all_pred_residual_fw)
            bw_flow_overall, bw_flow_agg, bw_residual_adjustment, bw_flow_affine = \
                self.aggregate_flow_with_residual(mask2, gt_bw_flow, all_pred_residual_bw)

            mask_fw_flow = self.detect_flow_changes_batch(gt_fw_flow)
            mask_bw_flow = self.detect_flow_changes_batch(gt_bw_flow)

            if not self.outlier_robust_loss:
                losses_fw = ((gt_fw_flow - fw_flow_overall) ** 2) * mask_fw_flow
                losses_fw = losses_fw.sum(dim=(1, 2, 3)) / (mask_fw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                losses_bw = ((gt_bw_flow - bw_flow_overall) ** 2) * mask_bw_flow
                losses_bw = losses_bw.sum(dim=(1, 2, 3)) / (mask_bw_flow.sum(dim=(1, 2, 3)) + 1e-6)
            else:
                losses_fw = ((((gt_fw_flow - fw_flow_overall).abs()).view(batch_size, -1)
                              + self.eps) ** self.q).mean(dim=1)
                losses_bw = ((((gt_bw_flow - bw_flow_overall).abs()).view(batch_size, -1)
                              + self.eps) ** self.q).mean(dim=1)

            individual_losses_fw.append(losses_fw)
            individual_losses_bw.append(losses_bw)

            _h, _w, flow, flow2 = get_norm_flow(lis1=gt_fw_flow, lis2=gt_bw_flow)
            flows['gt_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_overall, lis2=bw_flow_overall)
            flows['pred_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_agg, lis2=bw_flow_agg)
            flows['agg_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_residual_adjustment, lis2=bw_residual_adjustment)
            flows['residual_adj'].append(torch.cat([flow, flow2], dim=1))

            if fw_flow_affine is not None:
                _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_affine, lis2=bw_flow_affine)
                flows['affine_flow'].append(torch.cat([flow, flow2], dim=1))

        # ── topk selection (V2: configurable) ──────────────────────────
        total_losses_fw = torch.cat(individual_losses_fw)
        total_losses_bw = torch.cat(individual_losses_bw)
        total_losses = total_losses_fw + total_losses_bw

        sorted_losses, sorted_indices = torch.sort(total_losses)

        # Clamp topk to batch size to avoid index out of range
        k = min(self.topk, len(total_losses))
        selected_indices = sorted_indices[:k]

        selected_flow_loss = {
            'seg_fw': total_losses_fw[selected_indices].mean(),
            'seg_bw': total_losses_bw[selected_indices].mean()
        }
        selected_flow_loss['seg'] = selected_flow_loss['seg_fw'] + selected_flow_loss['seg_bw']

        selected_flows = {}
        for key, value in flows.items():
            if len(value) >= len(selected_indices):
                selected_flows[key] = [value[i] for i in selected_indices]
            else:
                selected_flows[key] = value.copy()

        return selected_flows, selected_flow_loss

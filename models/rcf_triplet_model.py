"""
RCFTripletModel: trains on 3 consecutive frames (frame_i, frame_j=i+1,
frame_k=i+2) per sample, computing genuine supervision for all THREE
pairwise relationships (i-j, j-k, i-k) instead of just one pair -- "option
B" from this session's 3-frame discussion (260720), as opposed to the
cheaper "3rd frame as auxiliary attention input only" option.

Reuses decode_head2 (mask), decode_head3 (residual), and decode_head
(FlowAggregationHeadWithResidualV2, the flow-reconstruction loss)
COMPLETELY UNMODIFIED -- none of them are hardcoded to any particular
frame identity; they only need "a mask pair + a flow pair + a residual
pair" to compute a loss, so the SAME instances are called 3 times, once
per pairwise relationship, and the 3 losses are summed. Only the
ORCHESTRATION (which frames pair with which, calling shared components 3x,
combining 3 losses) is new, living entirely in this one model class.

No DINO auxiliary loss here (RCFDinoModel's L_dino path is keyed off frame
0's captured mask logits inside the base forward_train, which this class
does not call at all -- deliberately out of scope for this first triplet
version, to keep the new orchestration logic self-contained and easy to
verify). Subclasses RCFSoftTissueModel only to reuse its __init__
(backbone/decode_head2/decode_head3/decode_head construction, tissue-role
channel bookkeeping) -- forward_train and forward are both fully replaced,
not extended.

Pairs with dataset/triplet_data.py (TripletVideoDataset) for data loading.
"""
import torch
import torch.nn.functional as F

from models.rcf_soft_tissue_model import RCFSoftTissueModel


class RCFTripletModel(RCFSoftTissueModel):
    def _residual_for_pair(self, all_feat, batch_size, idx_a, idx_b):
        """Reuses decode_head3 (FCNHead) unmodified -- same computation
        pred_separate_residual does for the im_num=2 case, generalized to
        pick an ARBITRARY pair of frame indices out of a >2-frame batch."""
        feats_pair = []
        for feat in all_feat:
            f = feat.unflatten(0, (batch_size, 3))          # [B, 3, C, H, W]
            pair = torch.cat([f[:, idx_a], f[:, idx_b]], dim=1)  # [B, 2C, H, W]
            feats_pair.append(pair)
        all_pred_residual = self._decode_head_forward(feats_pair, self.decode_head3)
        fw = all_pred_residual[:, :2 * self.num_classes]
        bw = all_pred_residual[:, 2 * self.num_classes:]
        return fw, bw

    def _loss_for_pair(self, imgs, all_pred_mask, all_feat, batch_size,
                       idx_a, idx_b, gt_fw_flow, gt_bw_flow, seq_names):
        """Reuses decode_head (FlowAggregationHeadWithResidualV2) unmodified
        -- constructs the exact [B,2,...] shapes it expects for an arbitrary
        pair (idx_a, idx_b) within the 3-frame batch."""
        imgs_pair = torch.stack([imgs[:, idx_a], imgs[:, idx_b]], dim=1)      # [B,2,C,H,W]
        mask_pair = torch.stack([all_pred_mask[:, idx_a], all_pred_mask[:, idx_b]], dim=1)  # [B,2,5,h,w]
        residual_fw, residual_bw = self._residual_for_pair(all_feat, batch_size, idx_a, idx_b)

        _, loss_flow = self.decode_head(
            imgs_pair, mask_pair, gt_fw_flow, gt_bw_flow, residual_fw, residual_bw,
            seq_names=seq_names,
        )
        return loss_flow['seg']

    def forward_train(self, imgs, seq_ids, seq_names, paths, flows, pl_masks=None, gaps=None):
        """flows: dict with 6 keys (flow_ij_fw, flow_ij_bw, flow_jk_fw,
        flow_jk_bw, flow_ik_fw, flow_ik_bw), each [B, 2, H, W] (already
        resized to crop resolution by the dataset/transform, NOT yet
        downsampled to mask_size -- that happens below, same as base
        forward_train does for the single-pair case)."""
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        assert im_num == 3, "RCFTripletModel requires exactly 3 frames"

        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)

        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        if self.allow_mask_resize and (all_pred_mask.shape[-2:] != self.mask_size):
            all_pred_mask = self.resize(all_pred_mask, self.mask_size)
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        def resize_flow(flow):
            # flow: [B, 2, H, W] -> [B, 1, 2, mask_h, mask_w] (flow_num=1, matches
            # _resize_gt_flow's expected input/FlowAggregationHead's expected shape)
            small = self._resize_gt_flow(flow, self.mask_size)
            return small.unsqueeze(1)

        loss_ij = self._loss_for_pair(imgs, all_pred_mask, all_feat, batch_size, 0, 1,
                                      resize_flow(flows['flow_ij_fw']), resize_flow(flows['flow_ij_bw']), seq_names)
        loss_jk = self._loss_for_pair(imgs, all_pred_mask, all_feat, batch_size, 1, 2,
                                      resize_flow(flows['flow_jk_fw']), resize_flow(flows['flow_jk_bw']), seq_names)
        loss_ik = self._loss_for_pair(imgs, all_pred_mask, all_feat, batch_size, 0, 2,
                                      resize_flow(flows['flow_ik_fw']), resize_flow(flows['flow_ik_bw']), seq_names)

        loss_warp_seg = (loss_ij + loss_jk + loss_ik) / 3.0
        loss = loss_warp_seg * self.w_seg

        self.train_iter += 1
        return {
            'loss_warp_seg': loss_warp_seg,
            'loss_warp_seg_ij': loss_ij.detach(),
            'loss_warp_seg_jk': loss_jk.detach(),
            'loss_warp_seg_ik': loss_ik.detach(),
            'loss': loss,
        }

    def forward(self, x, return_pred_vis_list=False):
        imgs = torch.stack(x['imgs'], dim=1)
        seq_ids, seq_names, paths = x['seq_ids'], x['seq_names'], x['paths']
        if self.training:
            flows = {k: torch.stack(x[k], dim=1)[:, 0] if isinstance(x[k], list) else x[k]
                    for k in ('flow_ij_fw', 'flow_ij_bw', 'flow_jk_fw', 'flow_jk_bw', 'flow_ik_fw', 'flow_ik_bw')}
            return self.forward_train(imgs, seq_ids, seq_names, paths, flows)
        else:
            return self.forward_eval(imgs, seq_ids, seq_names, paths, return_pred_vis_list=return_pred_vis_list)

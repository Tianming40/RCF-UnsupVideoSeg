"""
Diagnose where v102 (all-time champion, 139.77/140.12 sum) actually loses
accuracy: sparse boundary-only supervision, unreliable RAFT flow, or topk
batch-selection starving hard sequences of gradient -- vs. the alternative
hypothesis that single-frame ResNet features themselves are "corrupted" and
need denoising (the C-LaV-inspired direction under discussion).

Three diagnoses, read-only, no training:
  A. error rate vs. distance-to-boundary-supervision-region
  B. error rate vs. RAFT cycle-consistency confidence
  C. per-sequence flow-reconstruction loss (proxy for topk survival) vs.
     per-sequence eval error

Usage: python script/diagnose_v102_bottleneck.py [--n_eval 200] [--n_train_batches 200]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import main_tissue

CFG = "configs/instrument/rcf_cmc_grasp0_tissue_ft_v102.yaml"
CKPT = "saved_discrete_data/grasp0_tissue_ft_v102_260709_192435/epoch=27-step=10892.ckpt"
EVAL_ROOT = "/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_deinterlaced/eval_instrument"
TRAIN_FLOW_ROOT = "/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_5_10_merged"
VAL_SPLIT = os.path.join(EVAL_ROOT, "ImageSets/val.txt")
SCRATCH_DIR = "saved/diag_v102_scratch"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT = 400


def load_model():
    args = utils.load_args(CFG, cli_opts=[])
    args.rank = -1
    args.test = True
    args.multi_gpu = False
    args.resume_from_checkpoint = None
    args.pretrained_model = CKPT
    args.allow_overwriting_checkpoints_dir = True
    args.checkpoints_dir = SCRATCH_DIR
    model = main_tissue.TissueModel(args, trainer=None)
    model = model.cuda().eval()
    return model


def read_stems(split_file):
    stems = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "JPEGImages/<stem>/ <stem>.png"
            stem = line.split("/")[1]
            stems.append(stem)
    return stems


def resize_short_side(img, short=RESIZE_SHORT):
    w, h = img.size
    if w < h:
        new_w, new_h = short, int(round(h * short / w))
    else:
        new_h, new_w = short, int(round(w * short / h))
    return img.resize((new_w, new_h), Image.BILINEAR), (new_w / w, new_h / h)


def load_pair_and_ann(stem):
    img0_path = os.path.join(EVAL_ROOT, "JPEGImages", stem, f"{stem}.png")
    img1_path = os.path.join(EVAL_ROOT, "JPEGImages", stem, f"{stem}_1.png")
    ann_path = os.path.join(EVAL_ROOT, "Annotations", stem, f"{stem}.jpg")
    if not (os.path.exists(img0_path) and os.path.exists(img1_path) and os.path.exists(ann_path)):
        return None
    img0 = Image.open(img0_path).convert("RGB")
    img1 = Image.open(img1_path).convert("RGB")
    ann = Image.open(ann_path).convert("L")
    return img0, img1, ann


def find_flow(stem):
    for suffix in ("_g0", "_g5", "_g10"):
        fw_path = os.path.join(TRAIN_FLOW_ROOT, "Flows_NewCT", f"{stem}{suffix}", f"{stem}{suffix}_1.npy")
        bw_path = os.path.join(TRAIN_FLOW_ROOT, "BackwardFlows_NewCT", f"{stem}{suffix}", f"{stem}{suffix}_1.npy")
        if os.path.exists(fw_path) and os.path.exists(bw_path):
            fw = np.load(fw_path).astype(np.float32)  # [H, W, 2]
            bw = np.load(bw_path).astype(np.float32)
            return fw, bw
    return None


def to_tensor_img(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return t


def resize_flow(flow_hw2, new_h, new_w):
    # flow_hw2: [H, W, 2] numpy, pixel-displacement units. Rescale magnitude
    # by the same factor as spatial resize (unlike model._resize_gt_flow,
    # which resizes crops of matched scale -- here source/target resolutions
    # genuinely differ, so magnitude must be rescaled to stay in pixel units
    # of the NEW grid).
    h, w, _ = flow_hw2.shape
    t = torch.from_numpy(flow_hw2).permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    t[:, 0] *= (new_w / w)
    t[:, 1] *= (new_h / h)
    return t  # [1, 2, new_h, new_w]


@torch.no_grad()
def predict_mask(model, img0_tensor):
    feat = model.model.extract_feat(img0_tensor.cuda(), model.model.backbone2)
    pred = model.model._decode_head_forward(feat, model.model.decode_head2)
    pred = F.softmax(pred, dim=1)  # [1, C, h, w]
    return pred


def best_channel_iou(pred_full, ann_bin, ignore):
    # pred_full: [C, H, W] numpy prob, ann_bin: [H, W] {0,1}, ignore: [H,W] bool
    C = pred_full.shape[0]
    best_iou, best_c = -1, 0
    valid = ~ignore
    for c in range(C):
        p = (pred_full[c] > 0.35)
        inter = (p & (ann_bin == 1) & valid).sum()
        union = ((p | (ann_bin == 1)) & valid).sum()
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_iou, best_c = iou, c
    return best_c, best_iou


def diag_A_B(model, stems, n_eval):
    dist_bins = [0, 5, 15, 30, 60, 1e9]
    conf_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dist_err_sum = np.zeros(len(dist_bins) - 1)
    dist_err_cnt = np.zeros(len(dist_bins) - 1)
    conf_err_sum = np.zeros(len(conf_bins) - 1)
    conf_err_cnt = np.zeros(len(conf_bins) - 1)
    # GT-foreground-only variants: does recall on TRUE instrument pixels
    # degrade with distance from the flow-boundary supervision region? (the
    # whole-frame version above is dominated by the trivial background
    # majority class and can't distinguish "well supervised" from "mostly
    # easy background", since boundary-distance also just tracks proximity
    # to the true object edge, which is intrinsically the hardest region for
    # ANY segmenter, supervised or not.)
    dist_fg_miss_sum = np.zeros(len(dist_bins) - 1)
    dist_fg_cnt = np.zeros(len(dist_bins) - 1)
    conf_fg_miss_sum = np.zeros(len(conf_bins) - 1)
    conf_fg_cnt = np.zeros(len(conf_bins) - 1)
    per_seq_error = {}

    n_used = 0
    n_no_flow = 0
    for stem in stems:
        if n_used >= n_eval:
            break
        pair = load_pair_and_ann(stem)
        if pair is None:
            continue
        img0, img1, ann = pair
        flow_pair = find_flow(stem)
        if flow_pair is None:
            n_no_flow += 1
            continue
        fw_flow, bw_flow = flow_pair

        img0_r, scale0 = resize_short_side(img0)
        w_r, h_r = img0_r.size
        img0_t = to_tensor_img(img0_r)

        pred = predict_mask(model, img0_t)  # [1,C,h,w]
        pred_full = F.interpolate(pred, size=(h_r, w_r), mode="bilinear", align_corners=False)
        pred_full = pred_full[0].cpu().numpy()  # [C, h_r, w_r]

        ann_r = ann.resize((w_r, h_r), Image.NEAREST)
        ann_arr = np.asarray(ann_r)
        ignore = ann_arr == 128
        ann_bin = (ann_arr > 128).astype(np.uint8)

        best_c, best_iou = best_channel_iou(pred_full, ann_bin, ignore)
        pred_bin = (pred_full[best_c] > 0.35).astype(np.uint8)
        err_map = (pred_bin != ann_bin).astype(np.float32)

        fw_flow_r = resize_flow(fw_flow, h_r, w_r).cuda()
        bw_flow_r = resize_flow(bw_flow, h_r, w_r).cuda()
        fw_n = model.model.decode_head.norm_and_clamp_flow(fw_flow_r)
        bw_n = model.model.decode_head.norm_and_clamp_flow(bw_flow_r)

        boundary = model.model.decode_head.detect_flow_changes_batch(fw_n)  # [1,1,h,w], floored
        boundary_np = boundary[0, 0].cpu().numpy()
        # raw (unfloored) binary edge for the distance transform
        raw_edge = boundary_np > (model.model.decode_head.boundary_floor + 1e-6)
        if raw_edge.sum() == 0:
            dist_map = np.full_like(boundary_np, 1e9)
        else:
            dist_map = distance_transform_edt(~raw_edge)

        conf = model.model.decode_head._compute_cycle_conf(fw_n, bw_n)
        conf_np = conf[0, 0].cpu().numpy()

        valid = ~ignore
        fg = valid & (ann_bin == 1)
        miss_map = (pred_bin == 0).astype(np.float32)  # false negative indicator
        for i in range(len(dist_bins) - 1):
            m = valid & (dist_map >= dist_bins[i]) & (dist_map < dist_bins[i + 1])
            dist_err_sum[i] += err_map[m].sum()
            dist_err_cnt[i] += m.sum()
            mf = fg & (dist_map >= dist_bins[i]) & (dist_map < dist_bins[i + 1])
            dist_fg_miss_sum[i] += miss_map[mf].sum()
            dist_fg_cnt[i] += mf.sum()
        for i in range(len(conf_bins) - 1):
            m = valid & (conf_np >= conf_bins[i]) & (conf_np < conf_bins[i + 1])
            conf_err_sum[i] += err_map[m].sum()
            conf_err_cnt[i] += m.sum()
            mf = fg & (conf_np >= conf_bins[i]) & (conf_np < conf_bins[i + 1])
            conf_fg_miss_sum[i] += miss_map[mf].sum()
            conf_fg_cnt[i] += mf.sum()

        per_seq_error[stem] = 1.0 - best_iou
        n_used += 1

    print(f"\n[A/B] used {n_used} eval frames ({n_no_flow} skipped: no flow found)")
    print("\n[Diagnosis A] whole-frame pixel error rate vs. distance to boundary-supervision region:")
    for i in range(len(dist_bins) - 1):
        lo, hi = dist_bins[i], dist_bins[i + 1]
        rate = dist_err_sum[i] / max(dist_err_cnt[i], 1)
        print(f"  [{lo:>5.0f}, {hi:>7.0f}) px : err_rate={rate:.4f}  n_px={int(dist_err_cnt[i])}")

    print("\n[Diagnosis A'] TRUE-FOREGROUND-ONLY miss rate (false negative) vs. distance to boundary-supervision region:")
    for i in range(len(dist_bins) - 1):
        lo, hi = dist_bins[i], dist_bins[i + 1]
        rate = dist_fg_miss_sum[i] / max(dist_fg_cnt[i], 1)
        print(f"  [{lo:>5.0f}, {hi:>7.0f}) px : miss_rate={rate:.4f}  n_fg_px={int(dist_fg_cnt[i])}")

    print("\n[Diagnosis B] whole-frame pixel error rate vs. RAFT cycle-consistency confidence:")
    for i in range(len(conf_bins) - 1):
        lo, hi = conf_bins[i], conf_bins[i + 1]
        rate = conf_err_sum[i] / max(conf_err_cnt[i], 1)
        print(f"  conf [{lo:.1f}, {hi:.1f}) : err_rate={rate:.4f}  n_px={int(conf_err_cnt[i])}")

    print("\n[Diagnosis B'] TRUE-FOREGROUND-ONLY miss rate vs. RAFT cycle-consistency confidence:")
    for i in range(len(conf_bins) - 1):
        lo, hi = conf_bins[i], conf_bins[i + 1]
        rate = conf_fg_miss_sum[i] / max(conf_fg_cnt[i], 1)
        print(f"  conf [{lo:.1f}, {hi:.1f}) : miss_rate={rate:.4f}  n_fg_px={int(conf_fg_cnt[i])}")

    return per_seq_error


@torch.no_grad()
def diag_C(model, n_batches):
    loader = model.train_dataloader()
    per_seq_losses = {}
    n_done = 0
    for batch in loader:
        if n_done >= n_batches:
            break
        imgs = torch.stack(batch["imgs"], dim=1).cuda()
        gt_fw = torch.stack(batch["gt_fw_flows"], dim=1).cuda()
        gt_bw = torch.stack(batch["gt_bw_flows"], dim=1).cuda()
        seq_names = batch["seq_names"]
        gaps = batch["gap"]

        m = model.model
        B, im_num, C, h, w = imgs.shape
        img3 = imgs.reshape(B * im_num, C, h, w)
        all_feat = m.extract_feat(img3, m.backbone2)
        all_pred_mask = m._decode_head_forward(all_feat, m.decode_head2)
        if m.allow_mask_resize and all_pred_mask.shape[-2:] != m.mask_size:
            all_pred_mask = m.resize(all_pred_mask, m.mask_size)
        all_pred_residual_fw, all_pred_residual_bw = m.pred_separate_residual(all_feat, B, im_num)
        _, _, fh, fw_ = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(B, im_num, m.mask_layer, fh, fw_)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        gt_fw_r = m._resize_gt_flow(gt_fw.reshape(B * gt_fw.shape[1], *gt_fw.shape[2:]), m.mask_size)
        gt_bw_r = m._resize_gt_flow(gt_bw.reshape(B * gt_bw.shape[1], *gt_bw.shape[2:]), m.mask_size)
        gt_fw_r = gt_fw_r.view(B, gt_fw.shape[1], 2, *m.mask_size)
        gt_bw_r = gt_bw_r.view(B, gt_bw.shape[1], 2, *m.mask_size)

        pred_flows, _ = m.decode_head(imgs, all_pred_mask, gt_fw_r, gt_bw_r,
                                       all_pred_residual_fw, all_pred_residual_bw,
                                       seq_names=seq_names, gaps=gaps)
        gt = pred_flows["gt_flow"][0]   # [B,4,h,w] (normalised, see get_norm_flow)
        pr = pred_flows["pred_flow"][0]
        err = (gt - pr) ** 2
        per_sample_loss = err.mean(dim=(1, 2, 3)).cpu().numpy()

        for name, loss in zip(seq_names, per_sample_loss):
            base = name.split("_g")[0] if "_g" in name else name
            per_seq_losses.setdefault(base, []).append(float(loss))
        n_done += 1

    print(f"\n[C] scanned {n_done} training batches")
    seq_mean_loss = {k: float(np.mean(v)) for k, v in per_seq_losses.items() if len(v) >= 2}
    ranked = sorted(seq_mean_loss.items(), key=lambda x: -x[1])
    print("\n[Diagnosis C] top-20 persistently-high flow-reconstruction-loss sequences:")
    for stem, loss in ranked[:20]:
        print(f"  {stem:>24s}  mean_loss={loss:.4f}  n_samples={len(per_seq_losses[stem])}")
    return seq_mean_loss, ranked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_train_batches", type=int, default=200)
    args = p.parse_args()

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRATCH_DIR, "saved_eval"), exist_ok=True)
    os.makedirs(os.path.join(SCRATCH_DIR, "saved_eval_export"), exist_ok=True)

    print("Loading v102 champion checkpoint...")
    model = load_model()

    stems = read_stems(VAL_SPLIT)
    print(f"{len(stems)} eval stems found")

    per_seq_error = diag_A_B(model, stems, args.n_eval)
    seq_mean_loss, ranked = diag_C(model, args.n_train_batches)

    print("\n[Cross-check] overlap between diagnosis C's hardest sequences and diagnosis A/B's highest-error eval stems:")
    err_ranked = sorted(per_seq_error.items(), key=lambda x: -x[1])
    hard_c = set(k for k, _ in ranked[:30])
    hard_ab = set(k for k, _ in err_ranked[:30])
    overlap = hard_c & hard_ab
    print(f"  top-30 C ∩ top-30 A/B = {len(overlap)} / 30 sequences: {sorted(overlap)}")


if __name__ == "__main__":
    main()

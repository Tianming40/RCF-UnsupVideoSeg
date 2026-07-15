#!/usr/bin/env python3
"""
epistemic_vs_aleatoric_check.py

Cheap TTA-based epistemic-uncertainty proxy vs the model's own learned
aleatoric sigma (v96/v98's heteroscedastic head), for a handful of
instrument-heavy eval pairs.

Hypothesis: the current heteroscedastic loss models ALEATORIC uncertainty
only (Kendall & Gal 2017 style) — should be high where flow is genuinely
unlearnable (occlusion, specular highlight), low elsewhere. If sigma is
instead roughly flat while the true (epistemic) uncertainty is sharply
concentrated on the instrument, that means sigma isn't tracking real
difficulty and is instead diluting gradient roughly uniformly.

v1 of this script fed the FULL native-resolution frame directly through
the backbone. That's wrong: backbone2 (dilated ResNet) and decode_head2
are only ever run on 384x384 windows (see main.py _sliding_window_eval,
used by real eval whenever use_sliding_window=True). A raw full-frame
forward pass is off-distribution and produced garbage (a channel whose
only >0.5 pixels were a border artifact, not the instrument).

Fix (this version), two stages:
  1. Localize the instrument honestly: run the SAME sliding-window
     backbone2+decode_head2 inference real eval uses, over the whole
     frame, to get a trustworthy full-resolution mask. Pick the
     non-background channel with the SMALLER activated area (instrument
     is consistently the minority-area foreground class vs tissue here).
  2. Crop a single native 384x384 window centered on that instrument
     blob, and run the full pipeline (backbone2 + decode_head2 +
     decode_head3 + decode_head/sigma) on just that crop — 384 is
     exactly the resolution the whole network (incl. decode_head3's
     residual head and decode_head's sigma head) was trained at, so a
     single-window forward here is on-distribution and valid.
Epistemic proxy = per-pixel std of the (single-window) instrument
probability map across N TTA passes (photometric jitter only, no
retrain/dropout needed).

Usage:
    python tools/epistemic_vs_aleatoric_check.py \
        --config configs/instrument/rcf_cmc_grasp0_tissue_ft_v96.yaml \
        --ckpt saved/grasp0_tissue_ft_v96_260708_222643/epoch=35-step=8820.ckpt \
        --out saved/epistemic_check_v96
"""
import sys
sys.path.insert(0, '.')

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import utils.utils as utils
import models
import main_tissue  # noqa: F401  (side effect: registers RCFSoftTissueModel into `models`)

DEVICE = 'cuda'

CASES = [
    '96524110300300020859_g0', '96463700700100009347_g0', '96402182500100014427_g0',
    '96394327100100011837_g0', '964434955001E0006395_g0', '96463700700100005740_g0',
]

N_TTA = 12
WIN = 384
BG_CHANNELS = {0, 3, 4}
BORDER = 4  # px, in mask-head resolution units, excluded when looking for real blobs


def load_model(config_path, ckpt_path):
    args = utils.load_args(config_path, cli_opts=[])
    args.rank = -1
    model = models.__dict__[args.model_cls](args, **args.model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(DEVICE)
    return model


def load_pair(case, ds_root):
    d = f'{ds_root}/JPEGImages/{case}'
    files = sorted(os.listdir(d))
    pre = np.asarray(Image.open(f'{d}/{files[0]}').convert('RGB')).astype(np.float32)
    post = np.asarray(Image.open(f'{d}/{files[1]}').convert('RGB')).astype(np.float32)
    fw = np.load(f'{ds_root}/Flows_NewCT/{case}/{case}_1.npy').astype(np.float32)
    bw = np.load(f'{ds_root}/BackwardFlows_NewCT/{case}/{case}_1.npy').astype(np.float32)
    return pre, post, fw, bw


def sliding_window_mask(model, img_np, window_size=WIN, stride=192):
    """Reimplements main.py Model._sliding_window_eval for a SINGLE frame
    (mask segmentation is per-frame; only decode_head3/decode_head pair
    frames, decode_head2 doesn't need to). Returns [mask_layer, H, W] softmax."""
    H, W = img_np.shape[:2]
    t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    mask_layer = model.mask_layer
    pred_accum = torch.zeros(1, mask_layer, H, W, device=DEVICE)
    count_map = torch.zeros(1, 1, H, W, device=DEVICE)

    def positions(length):
        if length <= window_size:
            return [0]
        pts = list(range(0, length - window_size, stride))
        if not pts or pts[-1] + window_size < length:
            pts.append(length - window_size)
        return sorted(set(pts))

    with torch.no_grad():
        for y in positions(H):
            for x in positions(W):
                y2, x2 = min(y + window_size, H), min(x + window_size, W)
                crop = t[:, :, y:y2, x:x2]
                ch, cw = crop.shape[-2], crop.shape[-1]
                if ch < window_size or cw < window_size:
                    crop = F.pad(crop, (0, window_size - cw, 0, window_size - ch))
                feat = model.extract_feat(crop, model.backbone2)
                pred = model._decode_head_forward(feat, model.decode_head2)
                pred = F.softmax(pred, dim=1)
                pred = F.interpolate(pred, size=(window_size, window_size), mode='bilinear', align_corners=False)
                pred_accum[:, :, y:y2, x:x2] += pred[:, :, :ch, :cw]
                count_map[:, :, y:y2, x:x2] += 1
    return (pred_accum / count_map.clamp(min=1))[0].cpu().numpy()


def find_instrument_bbox(mask_full):
    """mask_full: [mask_layer, H, W] softmax probs (sliding-window, trustworthy).
    Returns (channel, (y0,y1,x0,x1)) for the smaller-area non-bg blob, or None."""
    H, W = mask_full.shape[1:]
    candidates = [c for c in range(mask_full.shape[0]) if c not in BG_CHANNELS]
    best = None
    for c in candidates:
        m = mask_full[c] > 0.5
        m[:BORDER] = m[-BORDER:] = m[:, :BORDER] = m[:, -BORDER:] = False
        area = m.sum()
        if area < 20:
            continue
        if best is None or area < best[2]:
            ys, xs = np.where(m)
            best = (c, (ys.min(), ys.max(), xs.min(), xs.max()), area)
    if best is None:
        return None
    c, (y0, y1, x0, x1), area = best
    return c, (y0, y1, x0, x1)


def crop_window(pre, post, fw, bw, bbox, win=WIN):
    H, W = pre.shape[:2]
    y0, y1, x0, x1 = bbox
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0c = int(np.clip(cy - win // 2, 0, max(0, H - win)))
    x0c = int(np.clip(cx - win // 2, 0, max(0, W - win)))
    y1c, x1c = y0c + win, x0c + win
    sl = (slice(y0c, y1c), slice(x0c, x1c))
    return pre[sl], post[sl], fw[sl], bw[sl], (y0c, x0c)


def to_tensor_imgs(pre, post, jitter=None):
    imgs = np.stack([pre, post], axis=0)
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    if jitter is not None:
        gain, bias, gamma = jitter
        t = t / 255.0
        t = torch.clamp(t * gain + bias, 0, 1) ** gamma
        t = t * 255.0
    return t.unsqueeze(0).to(DEVICE)  # [1, 2, 3, win, win]


def forward_once(model, imgs, fw_flow_t, bw_flow_t):
    batch_size, im_num, num_channels, _h, _w = imgs.shape
    img_3 = imgs.reshape(batch_size * im_num, num_channels, _h, _w)
    all_feat = model.extract_feat(img_3, model.backbone2)

    all_pred_mask = model._decode_head_forward(all_feat, model.decode_head2)
    if model.allow_mask_resize and (all_pred_mask.shape[-2:] != tuple(model.mask_size)):
        all_pred_mask = model.resize(all_pred_mask, model.mask_size)

    if model.separate_residual:
        all_pred_residual_fw, all_pred_residual_bw = model.pred_separate_residual(all_feat, batch_size, im_num)
    else:
        all_pred_residual_fw, all_pred_residual_bw = model.pred_joint_residual(
            all_feat[-1].unflatten(0, (batch_size, im_num)))

    _, _, _feat_h, _feat_w = all_pred_mask.shape
    all_pred_mask = all_pred_mask.view(batch_size, im_num, model.mask_layer, _feat_h, _feat_w)
    all_pred_mask = F.softmax(all_pred_mask, dim=2)

    gt_fw_flows = model.resize(fw_flow_t, model.mask_size).unsqueeze(1)
    gt_bw_flows = model.resize(bw_flow_t, model.mask_size).unsqueeze(1)

    pred_flows, _ = model.decode_head(imgs, all_pred_mask, gt_fw_flows, gt_bw_flows,
                                       all_pred_residual_fw, all_pred_residual_bw)

    sigma_fw = pred_flows['sigma_fw'][0] if pred_flows['sigma_fw'] else None
    return all_pred_mask.detach(), sigma_fw


def colorize(map01):
    m = np.clip(map01, 0, 1)
    r = np.clip(3 * m, 0, 1)
    g = np.clip(3 * m - 1, 0, 1)
    b = np.clip(3 * m - 2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def norm01(x):
    hi = np.percentile(x, 99.5)
    return np.clip(x / (hi + 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', default='/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_5_10_merged')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.config, args.ckpt)

    all_epi_in, all_epi_out, all_ale_in, all_ale_out = [], [], [], []

    for case in CASES:
        try:
            pre, post, fw, bw = load_pair(case, args.data_root)
        except FileNotFoundError as e:
            print(f"SKIP {case}: {e}")
            continue

        # ── stage 1: honest localization via real sliding-window inference ──
        mask_full = sliding_window_mask(model, pre)
        found = find_instrument_bbox(mask_full)
        if found is None:
            print(f"{case}: no non-bg blob found, skipping")
            continue
        inst_ch, bbox = found
        print(f"{case}: sliding-window found instrument ch{inst_ch} bbox={bbox}")

        # ── stage 2: single native-res 384 crop, full pipeline (valid resolution) ──
        pre_c, post_c, fw_c, bw_c, _ = crop_window(pre, post, fw, bw, bbox)
        H, W = pre_c.shape[:2]
        fw_t = torch.from_numpy(fw_c).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        bw_t = torch.from_numpy(bw_c).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        imgs0 = to_tensor_imgs(pre_c, post_c)
        with torch.no_grad():
            mask0, sigma_fw = forward_once(model, imgs0, fw_t, bw_t)
        if sigma_fw is None:
            print("Model has no heteroscedastic head — aborting.")
            return
        inst_mask0 = mask0[0, 0, inst_ch]

        tta_masks = []
        rng = np.random.default_rng(0)
        for _ in range(N_TTA):
            gain = float(rng.uniform(0.85, 1.15))
            bias = float(rng.uniform(-0.06, 0.06))
            gamma = float(rng.uniform(0.85, 1.15))
            imgs_j = to_tensor_imgs(pre_c, post_c, jitter=(gain, bias, gamma))
            with torch.no_grad():
                mask_j, _ = forward_once(model, imgs_j, fw_t, bw_t)
            tta_masks.append(mask_j[0, 0, inst_ch].cpu().numpy())
        tta_masks = np.stack(tta_masks, axis=0)
        epistemic = tta_masks.std(axis=0)
        aleatoric = sigma_fw[0].mean(dim=0).cpu().numpy()

        inst_prob_np = inst_mask0.cpu().numpy()
        inst_binary = inst_prob_np > 0.5

        if inst_binary.sum() > 10:
            epi_in = epistemic[inst_binary].mean()
            epi_out = epistemic[~inst_binary].mean()
            ale_in = aleatoric[inst_binary].mean()
            ale_out = aleatoric[~inst_binary].mean()
            all_epi_in.append(epi_in); all_epi_out.append(epi_out)
            all_ale_in.append(ale_in); all_ale_out.append(ale_out)
            print(f"  instrument-frac(in-384-crop)={inst_binary.mean():.2f}  "
                  f"epistemic(in/out)={epi_in:.4f}/{epi_out:.4f}  "
                  f"aleatoric(in/out)={ale_in:.4f}/{ale_out:.4f}")
        else:
            print(f"  instrument mask nearly empty in this crop, skipping stats")

        def up(x):
            t = torch.from_numpy(x)[None, None].float()
            t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
            return t[0, 0].numpy()

        panel = np.concatenate([
            pre_c.astype(np.uint8),
            colorize(up(inst_prob_np)),
            colorize(up(norm01(epistemic))),
            colorize(up(norm01(aleatoric))),
        ], axis=1)
        Image.fromarray(panel).save(f'{args.out}/{case}.png')

    print("\ncolumns: orig(384crop) | pred_instrument_prob | epistemic(TTA std) | aleatoric(learned sigma)")
    if all_epi_in:
        print(f"\nAGGREGATE over {len(all_epi_in)} cases:")
        print(f"  epistemic  in-instrument avg={np.mean(all_epi_in):.4f}  out avg={np.mean(all_epi_out):.4f}  ratio={np.mean(all_epi_in)/np.mean(all_epi_out):.2f}x")
        print(f"  aleatoric  in-instrument avg={np.mean(all_ale_in):.4f}  out avg={np.mean(all_ale_out):.4f}  ratio={np.mean(all_ale_in)/np.mean(all_ale_out):.2f}x")


if __name__ == '__main__':
    main()

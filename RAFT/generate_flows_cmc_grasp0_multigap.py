"""
RAFT flow (+confidence) generation for the new grasp0 multi-gap sequence
(CMC_grasp0_continuous_bwdif — pre, post_1..post_7, real consecutive frames
at increasing distance from pre, per the advisor's note; already
deinterlaced by tools/deinterlace_cmc_grasp0_multigap.py).

Generates flow for EVERY pairwise combination of the 8 frames per case
(C(8,2) = 28 pairs, fw+bw each = 56 flow fields per case), not just the
pre-anchored 7. Frame indices: 0=pre, 1=post_1(=post), 2=post_2, ...,
7=post_7. gap = j - i for a pair (i, j) with i < j.

Naming (clear, self-describing, greppable by gap or by specific frame
pair): for case <stem>, pair (i, j):
  Flows/<stem>/<stem>_f{i}t{j}_gap{j-i}.npy          -- forward flow i->j
  BackwardFlows/<stem>/<stem>_f{i}t{j}_gap{j-i}.npy  -- backward flow j->i
    (paired with the SAME (i,j) name as its forward counterpart, i.e. this
    is the reverse-direction flow of the SAME pair, not the forward flow
    of the reversed pair (j,i) — avoids ever having both f{i}t{j} and
    f{j}t{i} directories, which would be redundant.)
  FlowConf/<stem>/... , BackwardFlowConf/<stem>/...   -- matching confidence

Same fw/bw + confidence convention as generate_flows_cmc_bwdif.py: flow and
confidence saved in the SAME forward pass per direction (return_confidence
costs no extra RAFT compute).

Usage:
  python generate_flows_cmc_grasp0_multigap.py \
      --model models/raft-things.pth \
      --data_root /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_continuous_bwdif \
      --out_root /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp0_multigap_flows
"""

import sys
sys.path.append('core')

import argparse
import itertools
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(path):
    img = np.array(Image.open(path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_flow(flow_tensor, npy_path, png_path):
    flo = flow_tensor[0].permute(1, 2, 0).cpu().numpy()
    np.save(str(npy_path), flo.astype(np.float16))
    cv2.imwrite(str(png_path), flow_viz.flow_to_image(flo))


def save_conf(conf_tensor, npy_path):
    conf = conf_tensor[0, 0].cpu().numpy()   # [H, W], values in (0,1]
    np.save(str(npy_path), conf.astype(np.float16))


def frame_path(case_dir, idx):
    stem = case_dir.name
    return case_dir / (f"{stem}.png" if idx == 0 else f"{stem}_{idx}.png")


def main(args):
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    fw_root      = out_root / "Flows"
    bw_root      = out_root / "BackwardFlows"
    fw_conf_root = out_root / "FlowConf"
    bw_conf_root = out_root / "BackwardFlowConf"
    for d in (fw_root, bw_root, fw_conf_root, bw_conf_root):
        d.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f"Found {len(case_dirs)} cases in {data_root}")

    pairs = list(itertools.combinations(range(8), 2))  # 28 pairs
    print(f"{len(pairs)} pairs per case (C(8,2)), {len(pairs)*2} flow fields per case")

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module.to(DEVICE)
    model.eval()

    skipped = 0
    errors = 0
    for case_dir in tqdm(case_dirs, desc="Generating multigap flows+confidence"):
        stem = case_dir.name

        fw_out = fw_root / stem
        bw_out = bw_root / stem
        fw_conf_out = fw_conf_root / stem
        bw_conf_out = bw_conf_root / stem
        for d in (fw_out, bw_out, fw_conf_out, bw_conf_out):
            d.mkdir(exist_ok=True)

        for i, j in pairs:
            gap = j - i
            tag = f"{stem}_f{i}t{j}_gap{gap}"

            fw_npy = fw_out / f"{tag}.npy"
            bw_npy = bw_out / f"{tag}.npy"
            fw_conf_npy = fw_conf_out / f"{tag}.npy"
            bw_conf_npy = bw_conf_out / f"{tag}.npy"

            if fw_npy.exists() and bw_npy.exists() and fw_conf_npy.exists() and bw_conf_npy.exists():
                skipped += 1
                continue

            img_i_path = frame_path(case_dir, i)
            img_j_path = frame_path(case_dir, j)
            if not img_i_path.exists() or not img_j_path.exists():
                print(f"  [warn] {stem}: missing frame {i} or {j}, skipping pair")
                errors += 1
                continue

            try:
                image_i = load_image(img_i_path)
                image_j = load_image(img_j_path)
                padder = InputPadder(image_i.shape)
                image_i_p, image_j_p = padder.pad(image_i, image_j)

                with torch.no_grad():
                    _, fw_flow, fw_conf = model(image_i_p, image_j_p, iters=20, test_mode=True, return_confidence=True)
                    save_flow(fw_flow, fw_npy, fw_out / f"{tag}.png")
                    save_conf(fw_conf, fw_conf_npy)

                    _, bw_flow, bw_conf = model(image_j_p, image_i_p, iters=20, test_mode=True, return_confidence=True)
                    save_flow(bw_flow, bw_npy, bw_out / f"{tag}.png")
                    save_conf(bw_conf, bw_conf_npy)
            except Exception as e:
                print(f"  [error] {stem} pair ({i},{j}): {e}")
                errors += 1

    print(f"\nDone. Skipped {skipped} already-existing pairs. Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           required=True, help="RAFT checkpoint path")
    parser.add_argument("--data_root",       required=True, help="CMC_grasp0_continuous_bwdif root")
    parser.add_argument("--out_root",        required=True, help="output root for Flows/BackwardFlows/FlowConf/BackwardFlowConf")
    parser.add_argument("--small",           action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr",  action="store_true")
    args = parser.parse_args()
    main(args)

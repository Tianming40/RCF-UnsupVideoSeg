
"""
python generate_flows_cmc.py       --model models/raft-things.pth       --data_root /media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_from_raw
"""

import sys
sys.path.append('core')

import argparse
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


def main(args):
    data_root = Path(args.data_root)
    jpeg_root = data_root / "JPEGImages"
    fw_root   = data_root / "Flows_NewCT"
    bw_root   = data_root / "BackwardFlows_NewCT"
    fw_root.mkdir(exist_ok=True)
    bw_root.mkdir(exist_ok=True)

    seq_dirs = sorted(d for d in jpeg_root.iterdir() if d.is_dir())
    print(f"Found {len(seq_dirs)} sequences in {jpeg_root}")

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module.to(DEVICE)
    model.eval()

    skipped = 0
    for seq_dir in tqdm(seq_dirs, desc="Generating flows"):
        images = sorted(seq_dir.glob("*.png"))
        if len(images) < 2:
            print(f"  [warn] {seq_dir.name}: only {len(images)} frame(s), skipping")
            continue

        fw_out = fw_root / seq_dir.name
        bw_out = bw_root / seq_dir.name
        fw_out.mkdir(exist_ok=True)
        bw_out.mkdir(exist_ok=True)

        # gap=1 pairs only
        for img1_path, img2_path in zip(images[:-1], images[1:]):
            stem   = img2_path.stem
            fw_npy = fw_out / f"{stem}.npy"
            bw_npy = bw_out / f"{stem}.npy"

            if fw_npy.exists() and bw_npy.exists():
                skipped += 1
                continue

            image1 = load_image(img1_path)
            image2 = load_image(img2_path)
            padder = InputPadder(image1.shape)
            image1_p, image2_p = padder.pad(image1, image2)

            with torch.no_grad():
                _, fw_flow = model(image1_p, image2_p, iters=20, test_mode=True)
                save_flow(fw_flow, fw_npy, fw_out / f"{stem}.png")

                _, bw_flow = model(image2_p, image1_p, iters=20, test_mode=True)
                save_flow(bw_flow, bw_npy, bw_out / f"{stem}.png")

    print(f"\nDone. Skipped {skipped} already-existing pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           required=True, help="RAFT checkpoint path")
    parser.add_argument("--data_root",       required=True, help="CMC_grasp10_finetune root")
    parser.add_argument("--small",           action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr",  action="store_true")
    args = parser.parse_args()
    main(args)

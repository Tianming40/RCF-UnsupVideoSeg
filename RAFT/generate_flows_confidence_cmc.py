
"""
Same as generate_flows_cmc.py, but ALSO exports a per-pixel RAFT matching
confidence map (max softmax probability over the finest-level local
correlation window at the final GRU iteration — see RAFT._derive_confidence).

This is a separate script (not a modification of generate_flows_cmc.py) so
the original flow generation — which produced every Flows_NewCT/*.npy used
by v64 through v97 — stays untouched and fully reproducible. This script
only ADDS a new confidence output; the flow values it produces are bit-for-
bit identical to the original script's (same model call, same padding, same
iters=20), just requesting return_confidence=True on the same forward pass
(no extra RAFT compute — confidence is derived from a tensor already
computed for the GRU update).

python generate_flows_confidence_cmc.py --model models/raft-things.pth --data_root <...>
"""

import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(path):
    img = np.array(Image.open(path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_conf(conf_tensor, npy_path):
    conf = conf_tensor[0, 0].cpu().numpy()   # [H, W], values in (0,1]
    np.save(str(npy_path), conf.astype(np.float16))


def main(args):
    data_root = Path(args.data_root)
    jpeg_root = data_root / "JPEGImages"
    fw_conf_root = data_root / "FlowConf_NewCT"
    bw_conf_root = data_root / "BackwardFlowConf_NewCT"
    fw_conf_root.mkdir(exist_ok=True)
    bw_conf_root.mkdir(exist_ok=True)

    seq_dirs = sorted(d for d in jpeg_root.iterdir() if d.is_dir())
    print(f"Found {len(seq_dirs)} sequences in {jpeg_root}")

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module.to(DEVICE)
    model.eval()

    skipped = 0
    for seq_dir in tqdm(seq_dirs, desc="Generating flow confidence"):
        images = sorted(seq_dir.glob("*.png"))
        if len(images) < 2:
            continue

        fw_out = fw_conf_root / seq_dir.name
        bw_out = bw_conf_root / seq_dir.name
        fw_out.mkdir(exist_ok=True)
        bw_out.mkdir(exist_ok=True)

        for img1_path, img2_path in zip(images[:-1], images[1:]):
            stem = img2_path.stem
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
                _, fw_flow, fw_conf = model(image1_p, image2_p, iters=20, test_mode=True, return_confidence=True)
                save_conf(fw_conf, fw_npy)

                _, bw_flow, bw_conf = model(image2_p, image1_p, iters=20, test_mode=True, return_confidence=True)
                save_conf(bw_conf, bw_npy)

    print(f"\nDone. Skipped {skipped} already-existing pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           required=True, help="RAFT checkpoint path")
    parser.add_argument("--data_root",       required=True, help="CMC dataset root (must already have JPEGImages)")
    parser.add_argument("--small",           action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr",  action="store_true")
    args = parser.parse_args()
    main(args)

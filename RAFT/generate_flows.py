import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

DATASETS = [
    'instrument_dataset_1',
    'instrument_dataset_2',
    'instrument_dataset_3',
    'instrument_dataset_4',
    'instrument_dataset_5',
    'instrument_dataset_6',
    'instrument_dataset_7',
    'instrument_dataset_8',
]

#  data_medical/）
# gap=1: Flows_NewCT,         BackwardFlows_NewCT
# gap=2: Flows_NewCT2,        BackwardFlows_NewCT2
# gap=3: Flows_NewCT3,        BackwardFlows_NewCT3
FLOW_DIRS = {
    1: ('Flows_NewCT',  'BackwardFlows_NewCT'),
    2: ('Flows_NewCT2', 'BackwardFlows_NewCT2'),
    3: ('Flows_NewCT3', 'BackwardFlows_NewCT3'),
}


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_flow(flow_tensor, save_path_npy, save_path_png):

    flo = flow_tensor[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    flo_16bit = flo.astype(np.float16)
    np.save(save_path_npy, flo_16bit)

    flo_vis = flow_viz.flow_to_image(flo)
    cv2.imwrite(save_path_png, flo_vis)


def make_output_dirs(data_root, dataset_name):

    for gap, (fw_dir, bw_dir) in FLOW_DIRS.items():
        for d in [fw_dir, bw_dir]:
            os.makedirs(os.path.join(data_root, d, dataset_name), exist_ok=True)


def run_dataset(model, data_root, dataset_name, global_pbar):
  

    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    images = sorted(
        glob.glob(os.path.join(img_dir, '*.png')) +
        glob.glob(os.path.join(img_dir, '*.jpg'))
    )
    if len(images) == 0:
        print(f"[warn] No images found in {img_dir}, skipping.")
        return

    make_output_dirs(data_root, dataset_name)


    # gap=1: zip(images[:-1], images[1:])      终点帧编号: 0001..0224  共224对
    # gap=2: zip(images[:-2], images[2:])      终点帧编号: 0002..0224  共223对
    # gap=3: zip(images[:-3], images[3:])      终点帧编号: 0003..0224  共222对

    for gap, (fw_dir_name, bw_dir_name) in FLOW_DIRS.items():
        fw_out = os.path.join(data_root, fw_dir_name, dataset_name)
        bw_out = os.path.join(data_root, bw_dir_name, dataset_name)

        src_images = images[:-gap]
        dst_images = images[gap:]
        pairs = list(zip(src_images, dst_images))

        skipped = 0
        desc = f"{dataset_name} gap={gap}"
        pbar = tqdm(pairs, desc=desc, leave=False,
                    bar_format='{l_bar}{bar:25}{r_bar}')

        for imfile1, imfile2 in pbar:
            stem = os.path.splitext(os.path.basename(imfile2))[0]  # e.g. "0001"
            fw_npy = os.path.join(fw_out, stem + '.npy')
            fw_png = os.path.join(fw_out, stem + '.png')
            bw_npy = os.path.join(bw_out, stem + '.npy')
            bw_png = os.path.join(bw_out, stem + '.png')

            if os.path.exists(fw_npy) and os.path.exists(bw_npy):
                skipped += 1
                global_pbar.update(1)
                pbar.set_postfix(skipped=skipped, frame=stem)
                continue

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_p, image2_p = padder.pad(image1, image2)

            with torch.no_grad():
                # ：image1 → image2
                _, fw_flow = model(image1_p, image2_p, iters=20, test_mode=True)
                save_flow(fw_flow, fw_npy, fw_png)

                # image2 → image1
                _, bw_flow = model(image2_p, image1_p, iters=20, test_mode=True)
                save_flow(bw_flow, bw_npy, bw_png)

            global_pbar.update(1)
            pbar.set_postfix(skipped=skipped, frame=stem)

        pbar.close()


def main(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()

    data_root = args.data_root  # e.g. /path/to/data_medical

    total_pairs = len(DATASETS) * (224 + 223 + 222)
    print(f"Total pairs to process: {total_pairs} (across {len(DATASETS)} datasets × 3 gaps)")
    print(f"Output root: {data_root}\n")

    global_pbar = tqdm(total=total_pairs, desc='Overall', unit='pair',
                       bar_format='{l_bar}{bar:30}{r_bar}')

    for dataset_name in DATASETS:
        global_pbar.set_description(f'Overall [{dataset_name}]')
        run_dataset(model, data_root, dataset_name, global_pbar)

    global_pbar.close()
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     required=True, help='RAFT checkpoint path')
    parser.add_argument('--data_root', required=True, help='root dir containing JPEGImages/')
    parser.add_argument('--small',            action='store_true')
    parser.add_argument('--mixed_precision',  action='store_true')
    parser.add_argument('--alternate_corr',   action='store_true')
    args = parser.parse_args()
    main(args)

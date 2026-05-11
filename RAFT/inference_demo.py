import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'



def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# ========== Helper functions ==========
def ensure_dir(path):
    """Ensure that the directory exists."""
    os.makedirs(path, exist_ok=True)


def get_output_paths(gap):
    """Get output directory paths for forward and backward flows based on gap."""
    fw_dir = f'{OUTPUT_BASE}/Flows_NewCT2_gap{gap}/{SEQUENCE_NAME}'
    bw_dir = f'{OUTPUT_BASE}/BackwardFlows_NewCT2_gap{gap}/{SEQUENCE_NAME}'
    ensure_dir(fw_dir)
    ensure_dir(bw_dir)
    return fw_dir, bw_dir
# ======================================


# ========== Modified viz functions with configurable paths ==========
def viz(img, flo, count, gap=1):
    """Save forward flow as .npy and .png."""
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo_16bit = flo.astype(np.float16)
    
    fw_dir, _ = get_output_paths(gap)
    np.save(f'{fw_dir}/{count:04d}.npy', flo_16bit)
    
    # Map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(f'{fw_dir}/{count:04d}.png', flo)


def viz2(img, flo, count, gap=1):
    """Save backward flow as .npy and .png."""
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo_16bit = flo.astype(np.float16)
    
    _, bw_dir = get_output_paths(gap)
    np.save(f'{bw_dir}/{count:04d}.npy', flo_16bit)
    
    # Map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(f'{bw_dir}/{count:04d}.png', flo)
# =====================================================================


# ========== Core demo functions with gap parameter ==========
def demo(args, gap=1):
    """Compute forward optical flow with specified frame gap."""
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    count = 1
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        total = len(images) - gap
        print(f"\nComputing forward flow with gap={gap}, total pairs: {total}...")
        
        for i in range(len(images) - gap):
            image1 = load_image(images[i])
            image2 = load_image(images[i + gap])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, count, gap)
            
            if count % 50 == 0:
                print(f"  gap={gap} forward: {count}/{total}")
            count += 1
    print(f"gap={gap} forward flow completed, total: {count-1} files")


def demo2(args, gap=1):
    """Compute backward optical flow with specified frame gap."""
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    count = 1
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        total = len(images) - gap
        print(f"\nComputing backward flow with gap={gap}, total pairs: {total}...")
        
        for i in range(len(images) - gap):
            image1 = load_image(images[i])
            image2 = load_image(images[i + gap])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            viz2(image1, flow_up, count, gap)
            
            if count % 50 == 0:
                print(f"  gap={gap} backward: {count}/{total}")
            count += 1
    print(f"gap={gap} backward flow completed, total: {count-1} files")
# =================================================================


# ========== Wrapper function to compute all gaps at once ==========
def compute_all_flows(args):
    """Compute optical flows for all gaps (1, 2, 3) in one call."""
    print("=" * 50)
    print("Computing optical flows for all gaps (1, 2, 3)")
    print(f"Output directory: {OUTPUT_BASE}")
    print("=" * 50)
    
    for gap in [1, 2, 3]:
        print(f"\n{'='*30}")
        print(f"Processing gap={gap}")
        print(f"{'='*30}")
        demo(args, gap)
        demo2(args, gap)
    
    print("\n" + "=" * 50)
    print("All flows completed!")
    print(f"Forward flow directories: {OUTPUT_BASE}/Flows_NewCT2_gap*/{SEQUENCE_NAME}")
    print(f"Backward flow directories: {OUTPUT_BASE}/BackwardFlows_NewCT2_gap*/{SEQUENCE_NAME}")
    print("=" * 50)
# =================================================================


# ========== Legacy functions for backward compatibility ==========
def demo3(args):
    """Legacy function for gap=3 forward flow."""
    demo(args, gap=3)


def demo4(args):
    """Legacy function for gap=3 backward flow."""
    demo2(args, gap=3)


def demo5(args):
    """Legacy function for gap=2 forward flow."""
    demo(args, gap=2)


def demo6(args):
    """Legacy function for gap=2 backward flow."""
    demo2(args, gap=2)
# =================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--gaps', type=int, nargs='+', default=[1,2,3],
                        help="Frame gaps to compute (e.g., --gaps 1 2 3)")
    parser.add_argument('--output_base', type=str, 
                        default='/media/mitiadmin/Micron_7450_1/tianming/dataset/data_medical',
                        help="base output directory for flows")
    parser.add_argument('--sequence_name', type=str, default='instrument_dataset_1',
                        help="sequence/subfolder name")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # Override global variables with command line arguments
    global OUTPUT_BASE, SEQUENCE_NAME
    OUTPUT_BASE = args.output_base
    SEQUENCE_NAME = args.sequence_name

    # Set display environment variable to avoid Qt errors
    os.environ['DISPLAY'] = ''
    
    # Compute flows for specified gaps
    for gap in args.gaps:
        demo(args, gap)
        demo2(args, gap)
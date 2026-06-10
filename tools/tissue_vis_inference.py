

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "SemanticConstraintsAndMAA"))

# ── Register V2 and RCFSoftTissueModel ───────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
from models.rcf_soft_tissue_model import RCFSoftTissueModel
_models_pkg.RCFDinoModel       = RCFDinoModel        # type: ignore[attr-defined]
_models_pkg.RCFSoftTissueModel = RCFSoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFTissueModel     = RCFSoftTissueModel  # type: ignore[attr-defined]
RCFTissueModel = RCFSoftTissueModel                  # local alias for type hints below

from tools.maa_union_inference import load_config, load_checkpoint, build_dataset, to_device

import yaml, argparse as _ap, copy
from models.rcf_model import RCFModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_tissue_model(cfg: dict, output_dir: str = "/tmp") -> RCFTissueModel:
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=True, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    kwargs = copy.deepcopy(cfg["model_kwargs"])
    kwargs["allow_mask_resize"] = True
    return RCFTissueModel(args=fake_args, **kwargs)


def color_overlay(img_01: torch.Tensor, mask: torch.Tensor, color: tuple) -> torch.Tensor:
    """
    img_01 : [3, H, W] in [0,1]
    mask   : [H, W]    in [0,1]  (soft or binary)
    color  : (R, G, B) each in [0,1]
    Returns overlaid [3, H, W].
    """
    out = img_01.clone()
    m = mask.unsqueeze(0)                    # [1, H, W]
    c = torch.tensor(color, device=img_01.device).view(3, 1, 1)
    out = out * (1 - m * 0.6) + c * (m * 0.6)
    return out.clamp(0, 1)


def run_inference(model, dataset, out_dir: Path,
                  tissue_channel: int,
                  instrument_channels: tuple,
                  pos_th: float = 0.35,
                  workers: int = 4):

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=False)

    model.eval()
    print(f"Running inference on {len(dataset)} frames → {vis_dir}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch = to_device(batch)
            imgs = torch.stack(batch["imgs"], dim=1)      # [1, 1, 3, H, W]

            pred_masks = model.forward_eval(
                imgs, batch["seq_ids"], batch["seq_names"], batch["paths"])
            # pred_masks: [1, C, fH, fW]  (already softmax'd)

            B, C, fH, fW = pred_masks.shape

            # Original image — use native resolution (no downscale)
            orig = imgs[0, 0]                              # [3, Himg, Wimg]
            orig_01 = ((orig.cpu() + 2.0) / 4.0).clamp(0, 1)
            H, W = orig_01.shape[1], orig_01.shape[2]     # display = original size
            orig_r = orig_01

            # Resize masks to display resolution
            masks_r = F.interpolate(pred_masks.cpu(), (H, W),
                                    mode="bilinear", align_corners=False)[0]  # [C, H, W]

            # Instrument union mask
            instr_prob = masks_r[instrument_channels[0]]
            for c in instrument_channels[1:]:
                instr_prob = torch.max(instr_prob, masks_r[c])
            instr_bin = (instr_prob > pos_th).float()

            # Tissue mask
            tissue_prob = masks_r[tissue_channel]
            tissue_bin  = (tissue_prob > pos_th).float()

            # ── Build grid columns ────────────────────────────────────────────
            cols = [orig_r]

            # Ch 0-4 grayscale masks
            for c in range(C):
                cols.append(masks_r[c:c+1].repeat(3, 1, 1))

            # Instrument overlay (red)
            cols.append(color_overlay(orig_r, instr_bin, (1.0, 0.2, 0.2)))

            # Tissue overlay (blue)
            cols.append(color_overlay(orig_r, tissue_prob, (0.3, 0.6, 1.0)))

            # Dual overlay: instrument red + tissue blue (on same image)
            dual = orig_r.clone()
            dual = color_overlay(dual, instr_bin,  (1.0, 0.2, 0.2))
            dual = color_overlay(dual, tissue_prob, (0.3, 0.6, 1.0))
            cols.append(dual)

            grid = torch.stack(cols, dim=0)               # [N, 3, H, W]

            seq_name   = batch["seq_names"][0]
            frame_name = batch["paths"][0][0].split("/")[-1][:-4]
            save_path  = vis_dir / f"{seq_name}_{frame_name}.jpg"
            # nrow=N → 所有列横向排列成一行
            torchvision.utils.save_image(grid, str(save_path), nrow=len(cols), padding=2)

    print(f"Done → {vis_dir}  ({len(dataset)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--split",    default="ImageSets/val.txt")
    p.add_argument("--tissue_channel",      type=int,          default=1)
    p.add_argument("--instrument_channels", type=int, nargs="+", default=[0, 2, 3])
    p.add_argument("--pos_th",   type=float, default=0.35)
    p.add_argument("--workers",  type=int,   default=4)
    args = p.parse_args()

    cfg   = load_config(args.config)
    model = build_tissue_model(cfg, output_dir=args.output)
    model = load_checkpoint(model, args.ckpt)
    model = model.to(DEVICE)
    model.args.eval_save = False

    dataset = build_dataset(cfg, split_override=args.split)

    run_inference(
        model, dataset, Path(args.output),
        tissue_channel=args.tissue_channel,
        instrument_channels=tuple(args.instrument_channels),
        pos_th=args.pos_th,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

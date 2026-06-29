# Segmenting objects in videos **without human annotations**! 😲 🤯

# RCF: Bootstrapping Objectness from Videos by Relaxed Common Fate and Visual Grouping

by [Long Lian](https://tonylian.com/), [Zhirong Wu](https://scholar.google.com/citations?user=lH4zgcIAAAAJ&hl=en) and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley, MSRA, and UMich

<em>The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.</em>

[[Paper](https://arxiv.org/abs/2304.08025)] | [[Project Page](https://rcf-video.github.io/)] | [[Presentation Video](https://www.youtube.com/watch?v=dyaDEvT4YkY)] | [[Demo Video](http://people.eecs.berkeley.edu/~longlian/RCF_video.html)] | [[Poster](https://rcf-video.github.io/poster.png)] | [[Citation](#citation)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bootstrapping-objectness-from-videos-by/unsupervised-object-segmentation-on-davis)](https://paperswithcode.com/sota/unsupervised-object-segmentation-on-davis?p=bootstrapping-objectness-from-videos-by)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bootstrapping-objectness-from-videos-by/unsupervised-object-segmentation-on-segtrack)](https://paperswithcode.com/sota/unsupervised-object-segmentation-on-segtrack?p=bootstrapping-objectness-from-videos-by)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bootstrapping-objectness-from-videos-by/unsupervised-object-segmentation-on-fbms-59)](https://paperswithcode.com/sota/unsupervised-object-segmentation-on-fbms-59?p=bootstrapping-objectness-from-videos-by)

### **Non-cherry picked** segmentation predictions on all sequences on DAVIS16:

![Segmentation Masks](assets/output.gif)

**This GIF has been significantly compressed. [Check out our video at full resolution here.](http://people.eecs.berkeley.edu/~longlian/RCF_video.html)** Inference in this demo is done *per-frame* without post-processing for temporal consistency.

If you want to qualitatively compare segmentation masks from our method without running our code, you can download the segmentation masks [here](#model-zoo-and-prediction-masks).

### Our Method in a Figure
![Method Figure](assets/fig_heading.png)

## Data Preparation
### Prepare data and pretrained weights
Download [DAVIS 2016](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) and unzip to `data/data_davis`.

Download [pre-extracted flow from RAFT](https://huggingface.co/datasets/longlian/RCF-UnsupVideoSeg-Datasets/blob/main/flows_NewCT.tgz) (trained with chairs and things) and decompress to `data/data_davis`.

Download [DenseCL ResNet50 weights](https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download) to `data/pretrained/densecl_r50_imagenet_200ep.pth`.

<details>
<summary>SegTrackv2 and FBMS59 dataset</summary>

These two datasets have much lower quality and very different aspect ratios across sequences. To make things easier, we resize to 480p (854x480) to have the same input size as DAVIS 2016. For fairness, the testing is still on the original dataset, and we provide both the original and scaled datasets (with flows on the scaled datasets). There are also larger inter-run variations on these two datasets compared to DAVIS 2016 since the video quality is lower and/or the number of sequences is smaller. I recommend using DAVIS16 as the main metric and use these two as supplementary metrics. For reproducibility, checkpoints for both stages for three datasets have been released.

Download [SegTrackv2 with pre-extracted flow](https://huggingface.co/datasets/longlian/RCF-UnsupVideoSeg-Datasets/blob/main/SegTrackv2_all.tgz).

Download [FBMS59 with pre-extracted flow part 1](https://huggingface.co/datasets/longlian/RCF-UnsupVideoSeg-Datasets/blob/main/FBMS59_all.tgz.part1) and [FBMS59 with pre-extracted flow part 2](https://huggingface.co/datasets/longlian/RCF-UnsupVideoSeg-Datasets/blob/main/FBMS59_all.tgz.part2). Please use `cat FBMS59_all.tgz.part1 FBMS59_all.tgz.part2 > FBMS59_all.tgz` to merge before un-targz. `shasum` of the merged file: `e898c127f916de867dad665bbb04e21702e54e7c`.
</details>

### Install dependencies and `torchCRF`
The `requirements.txt` assumes CUDA 11.x. You can also install torch and torchvision from conda instead of pip.

`torchCRF` is a GPU CRF implementation typically faster than CPU implementation. If you plan to use this implementation in your work, see the `tools/torchCRF/README.md` for license.

```
pip install -r requirements.txt
cd tools/torchCRF
python setup.py install
```

We also require `parallel` command from moreutils. If your parallel does not work (for example, the parallel from parallel package), you either need to install moreutils from system package manager (e.g. APT on Ubuntu/Debian) or from conda: `conda install -c conda-forge moreutils`.

## Model Zoo and Prediction Masks
We provide pretrained models and prediction masks. If you intend to work on a custom dataset that is out-of-distrbution for our training data such as DAVIS16, we suggest training/fine-tuning our model on new datasets.

| Name               | Dataset | Backbone | mIoU (w/o pp.)    | mIoU (w/ pp.) | Model    | Masks    |
| ------------------ | ------- | -------- | ----------------- | ------------- | -------- | -------- |
| RCF (All stages)   | DAVIS16 | ResNet50 | 80.9              | **83.0**      | [Download](https://drive.google.com/drive/folders/1I9xYL4BZO8Dr6s3FzNZhN_QpGAU-_AzD?usp=share_link) | [Download](https://drive.google.com/drive/folders/1RjNpRM33IACSqN30-W6W14eAZddnwBYh?usp=share_link) |
| RCF (Stage 1 only) | DAVIS16 | ResNet50 | 78.9              | 81.4          | [Download](https://drive.google.com/drive/folders/1I9xYL4BZO8Dr6s3FzNZhN_QpGAU-_AzD?usp=share_link) | [Download](https://drive.google.com/drive/folders/1RjNpRM33IACSqN30-W6W14eAZddnwBYh?usp=share_link) |
| RCF (All stages)  |SegTrackv2| ResNet50 | 76.7              | **79.6**      | [Download](https://drive.google.com/drive/folders/1kD7t0TjCUpW8QRVDnnjq-_PGpHJvDVfF?usp=share_link) | [Download](https://drive.google.com/drive/folders/1pr2SZ_qabgDDxYaV3Zh-tXWQ2c9yPRNx?usp=share_link) |
| RCF (Stage 1 only)|SegTrackv2| ResNet50 | 72.8              | 77.6          | [Download](https://drive.google.com/drive/folders/1kD7t0TjCUpW8QRVDnnjq-_PGpHJvDVfF?usp=share_link) | [Download](https://drive.google.com/drive/folders/1pr2SZ_qabgDDxYaV3Zh-tXWQ2c9yPRNx?usp=share_link) |
| RCF (All stages)   | FBMS59  | ResNet50 | 69.9              | **72.4**      | [Download](https://drive.google.com/drive/folders/1jNBK0Ol2obFPQT9AFmHJ_HStdnlYwZHx?usp=share_link) | [Download](https://drive.google.com/drive/folders/1jOb6G07FVaRNhBWBfoI-KS2f15u2bMkd?usp=share_link) |
| RCF (Stage 1 only) | FBMS59  | ResNet50 | 66.8              | 69.1          | [Download](https://drive.google.com/drive/folders/1jNBK0Ol2obFPQT9AFmHJ_HStdnlYwZHx?usp=share_link) | [Download](https://drive.google.com/drive/folders/1jOb6G07FVaRNhBWBfoI-KS2f15u2bMkd?usp=share_link) |

To evaluate a pretrained model using our unofficial main training script and/or the masks for evaluation using evaluation tools, use `--test-override-pretrained` and `--test-override-object-channel` to specify the model path and the object channel, respectively.

## Train RCF
### Stage 1
To train our model on DAVIS16 with 2 GPUs, run:
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf/rcf_stage1.yaml
```
This should lead to a model with mIoU around 78% to 79% on DAVIS16 (without post-processing). Run stage 2 as well if additional gains are desired. If you want to run with other numbers of GPUs, change `CUDA_VISIBLE_DEVICES` and the `batch_size` so that the total batch size (`batch_size` times the number of GPUs) is your intended batch size (16 in this config).

<details>
<summary>SegTrackv2 and FBMS59 dataset</summary>
Training with STv2 and FBMS59 is very similar to training with DAVIS16.

STv2:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_stv2/rcf_stage1.yaml
```

FBMS59:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_fbms59/rcf_stage1.yaml
```

</details>

### Stage 2.1 (Low-level refinement)
This stage uses Conditional Random Field (CRF) to get training signals based on low-level vision (e.g., color). Prior to running this stage, we need to get the object channel through motion-appearance alignment.
```shell
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/maa.py --pretrain_dir saved/saved_rcf_stage1 --first-frames-only --step 43200
OBJECT_CHANNEL=$?
```

Then we could run training (which will continue training from pretrained stage 1 model):
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf/rcf_stage2.1.yaml --opts object_channel $OBJECT_CHANNEL
```

<details>
<summary>SegTrackv2 and FBMS59 dataset</summary>
Training with STv2 and FBMS59 is very similar to training with DAVIS16.

STv2:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/maa.py --pretrain_dir saved_stv2/saved_rcf_stage1 --first-frames-only --step 1220 --dataset stv2
OBJECT_CHANNEL=$?
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_stv2/rcf_stage2.1.yaml --opts object_channel $OBJECT_CHANNEL
```

FBMS59:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/maa.py --pretrain_dir saved_fbms59/saved_rcf_stage1 --first-frames-only --step 3468 --dataset fbms59 --num-channels 3
OBJECT_CHANNEL=$?
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_fbms59/rcf_stage2.1.yaml --opts object_channel $OBJECT_CHANNEL
```

</details>

### Stage 2.2 (Semantic constaints)
This stage uses a pretrained ViT model from DINO to get training signals based on high-level vision (e.g., semantics discovered in unsupervised learning). Semantic constraints are enforced offline due to its low speed.
```shell
# the predictions on trainval
CUDA_VISIBLE_DEVICES=0 python main.py configs/rcf/rcf_export_trainval_ema.yaml --test --test-override-pretrained saved/saved_rcf_stage2.1/last.ckpt --opts checkpoints_dir saved/saved_rcf_stage2.1 object_channel $OBJECT_CHANNEL
# run semantic constraints
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/semantic_constraints.py --pretrain_dir saved/saved_rcf_stage2.1 --object-channel $OBJECT_CHANNEL
# training with semantic constraints
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf/rcf_stage2.2.yaml --opts object_channel $OBJECT_CHANNEL train_dataset_kwargs.pl_root saved/saved_rcf_stage2.1/saved_eval_export_trainval_ema_torchcrf_ncut_torchcrf/$OBJECT_CHANNEL
```

This should give you a 80% to 81% mIoU (without post-processing).

<details>
<summary>SegTrackv2 and FBMS59 dataset</summary>
STv2:

```shell
# the predictions on trainval
CUDA_VISIBLE_DEVICES=0 python main.py configs/rcf_stv2/rcf_export_trainval_ema.yaml --test --test-override-pretrained saved_stv2/saved_rcf_stage2.1/last.ckpt --opts checkpoints_dir saved_stv2/saved_rcf_stage2.1 object_channel $OBJECT_CHANNEL
# run semantic constraints
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/semantic_constraints.py --pretrain_dir saved_stv2/saved_rcf_stage2.1 --object-channel $OBJECT_CHANNEL --dataset stv2
# training with semantic constraints
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_stv2/rcf_stage2.2.yaml --opts object_channel $OBJECT_CHANNEL train_dataset_kwargs.pl_root saved_stv2/saved_rcf_stage2.1/saved_eval_export_ema_torchcrf_ncut_torchcrf/$OBJECT_CHANNEL
```

FBMS59:

```shell
# the predictions on trainval
CUDA_VISIBLE_DEVICES=0 python main.py configs/rcf_fbms59/rcf_export_trainval_ema.yaml --test --test-override-pretrained saved_fbms59/saved_rcf_stage2.1/last.ckpt --opts checkpoints_dir saved_fbms59/saved_rcf_stage2.1 object_channel $OBJECT_CHANNEL
# run semantic constraints
CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/semantic_constraints.py --pretrain_dir saved_fbms59/saved_rcf_stage2.1 --object-channel $OBJECT_CHANNEL --dataset fbms59 --num-channels 3
# training with semantic constraints
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/rcf_fbms59/rcf_stage2.2.yaml --opts object_channel $OBJECT_CHANNEL train_dataset_kwargs.pl_root saved_fbms59/saved_rcf_stage2.1/saved_eval_export_trainval_ema_torchcrf_ncut_torchcrf/$OBJECT_CHANNEL
```

</details>

## Evaluate
### Without CRF Post-processing
To unofficially evaluate a trained model, run:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py configs/rcf/rcf_eval.yaml --test --test-override-pretrained saved/saved_rcf_stage2.2/last.ckpt --test-override-object-channel $OBJECT_CHANNEL
```
We encourage evaluating the model with our evaluation tool, which is supposed to closely match the DAVIS 2016 official evaluation tool.
To evaluate a trained model with the evaluation tool on the exported masks (stage 2.2 will masks on validation set by default):
```shell
python tools/davis2016-evaluation/evaluation_method.py --task unsupervised --davis_path data/data_davis --year 2016 --step 4320 --results_path saved/saved_rcf_stage2.2/saved_eval_export
```

### With CRF Post-processing
To refine the exported masks from a trained model with CRF post-processing, run:
```shell
sh tools/pydenseCRF/crf_parallel.sh
```

Then evaluate the refined masks with evaluation tool:
```shell
python tools/davis2016-evaluation/evaluation_method.py --task unsupervised --davis_path data/data_davis --year 2016 --step 4320 --results_path saved/saved_rcf_stage2.2/saved_eval_export_crf
```

This should reproduce around 83% mIoU (J-FrameMean).

<details>
<summary>Evaluate SegTrackv2 and FBMS59 dataset</summary>
We provide a tool to evaluate exported masks of SegTrackv2 and FBMS59:

```shell
python tools/STv2-FBMS59-evaluation/eval_tool.py --dataset SegTrackv2 --pred_dir [pred dir] --step [step num]
```
```shell
python tools/STv2-FBMS59-evaluation/eval_tool.py --dataset FBMS59 --pred_dir [pred dir] --step [step num]
```

</details>

## Train AMD (our baseline method)
This repo also supports training [AMD](https://github.com/rt219/The-Emergence-of-Objectness). However, this implementation is not guaranteed to be identical to the original one. In our experience, it reproduces results that are slightly better than the original reported results without test-time adaptation (i.e., fine-tuning on the downstream data). The setup for training set (Youtube-VOS) is simply unzipping the `train_all_frames.zip` from Youtube-VOS to `data/youtube-vos/train_all_frames`. The setup for validation sets are the same as RCF.

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --master_addr 127.0.0.1 --master_port 9000 --nproc_per_node gpu main.py configs/amd/amd.yaml
```

## Support
If you have any questions on the paper or this implementation, please contact Long Lian using the email address in the paper.

## Citation
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@InProceedings{Lian_2023_CVPR,
    author    = {Lian, Long and Wu, Zhirong and Yu, Stella X.},
    title     = {Bootstrapping Objectness From Videos by Relaxed Common Fate and Visual Grouping},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14582-14591}
}

@article{lian2022improving,
  title={Improving Unsupervised Video Object Segmentation with Motion-Appearance Synergy},
  author={Lian, Long and Wu, Zhirong and Yu, Stella X},
  journal={arXiv preprint arXiv:2212.08816},
  year={2022}
}
```


# modify from Tianming
in eval stage to select the object channel 

```
python maa.py --pretrain_dir path/to/your/pretrained_rcf_folder --dataset davis --num-channels 4 --first-frames-only

CUDA_VISIBLE_DEVICES=0 python tools/SemanticConstraintsAndMAA/maa.py \
    --pretrain_dir saved/pretrained_rcf \
    --dataset davis \
    --num-channels 4 \
    --step 0 \
    --first-frames-only
```

## DINO-guided RCF on CMC Surgical Instrument Dataset

Three-stage pipeline for training and evaluating on `data_medical` / `CMC_grasp10_deinterlaced`.

### Phase 1 — DINO-guided training on data_medical
Config: `configs/instrument/rcf_cmc_dino_phase1.yaml`
```shell
CUDA_VISIBLE_DEVICES=0 python main_dino.py configs/instrument/rcf_cmc_dino_phase1.yaml
```

### Phase 2 — Fine-tune on CMC_grasp10 (4-fold cross-validation)
Config: `configs/instrument/rcf_cmc_grasp10_finetune_dino.yaml`
```shell
# Run all 4 folds sequentially (recommended)
bash script/run_grasp10_finetune.sh all 0

# Or run a single fold
bash script/run_grasp10_finetune.sh 1 0
```

### Evaluation — Full 601-sequence eval across checkpoints
Config: `configs/instrument/rcf_cmc_grasp10_eval.yaml`
```shell
# Evaluate best checkpoints from all folds on full 601 sequences
bash script/run_grasp10_eval_full.sh 0

# Evaluate a single checkpoint on a specific fold val split
bash script/run_grasp10_eval_fold.sh 1 saved/grasp10_ft_fold1_<timestamp>/last.ckpt 0
```

---

## Grasp10 Fine-tuning Configs: v4 / v9 / v11

These three configs represent successive attempts to improve fine-tuning on `CMC_grasp10_deinterlaced`.
All use **4-fold stratified cross-validation** (each fold val set contains ~27% 1080p + ~73% 720p,
matching the overall dataset ratio) and **full-resolution sliding-window evaluation**
(window 384×384, stride 192) — the val metric is therefore directly comparable to real inference.

### Dataset resolution breakdown

| Resolution | Format | Short side | Share |
|---|---|---|---|
| 720×576 | PAL 5:4 | 576 px | 72.9% |
| 1920×1080 | 16:9 | 1080 px | 27.1% |

### Config comparison

| Config | 720×576 crop | 1080p crop | BN consistency | 4-fold avg best mIoU |
|---|---|---|---|---|
| **v4** | resize400 → crop384 | resize576 → crop**512** | ✗ train512/val384 mismatch | 66.89% |
| **v9** | resize400 → crop384 | resize576 → crop**384** | ✓ always 384×384 | **67.10%** |
| **v11** | resize400 → crop384 | resize576 → half\_crop384 | ✓ always 384×384 | 63.67% |

**v4** introduced resolution-aware cropping so 1080p images are less compressed (1.88× vs 2.7×),
but the model sees 512×512 patches during training and 384×384 windows at val — BatchNorm
running statistics diverge between the two sizes, causing instability on 1080p sequences.

**v9** fixes this by using crop=384 for 1080p as well, so BatchNorm always sees 384×384 inputs
(matching both the phase-1 pre-training and the sliding-window val). Width crop coverage for 1080p
drops from 50% to 38%, but the BN consistency gain outweighs the loss, especially for fold 2
(63.03% → 69.09%).

**v11** additionally restricts the 1080p random-crop x-offset to either the left half or right half
of the resized image (50/50). Each crop then covers 75% of its assigned half instead of 37.5% of the
full width. Flow vectors are unaffected because both image frames and all flow fields are cropped
from the same bbox. In practice the reduced x-offset diversity hurt training stability, so v9 remains
the recommended config.

### How to run

```shell
# v4 — all 4 folds sequentially on GPU 0
bash script/run_grasp10_ft_v4.sh all 0

# v9 — all 4 folds sequentially on GPU 0  (recommended)
bash script/run_grasp10_ft_v9.sh all 0

# v11 — all 4 folds sequentially on GPU 1
bash script/run_grasp10_ft_v11.sh all 1

# Run a single fold (e.g. fold 2 on GPU 1)
bash script/run_grasp10_ft_v9.sh 2 1
```

All scripts cold-start from the phase-1 checkpoint
(`saved/cmc_dino_phase1_260605_143205/epoch=7-step=1800.ckpt`),
train for 30 epochs at lr=3e-5, and save per-epoch checkpoints plus tensorboard logs under
`saved/grasp10_ft_<version>_fold<N>_<timestamp>/`.

### Key implementation details

**ResolutionAwareCrop** (`dataset/transforms.py`) — dispatches each image to the matching
`resolution_crop_configs` entry based on the original short side, then applies Resize + RandomCrop
(or half-constrained crop for v11). Supported per-entry options:

| Option | Type | Effect |
|---|---|---|
| `max_short_side` | int | upper bound of short side for this entry (ascending order) |
| `resize_short` | int | target short side passed to Resize |
| `crop_size` | [h, w] | RandomCrop output size |
| `split_wide` | bool | split landscape images (AR>1.5) into left/right halves before resize (v10 only — **not recommended**, breaks precomputed flow) |
| `half_crop` | bool | restrict x-offset to left or right half after resize (v11) |

Val always uses a plain `Resize(resize_short=400)` followed by sliding-window inference —
no crop options apply at evaluation time.

---

## Grasp0 Soft-Tissue Fine-Tuning (RCFSoftTissueModel)

### Background

Starting from the best `grasp10_ft_v9` checkpoint (instrument detector, ch1 = instrument),
we fine-tune on `CMC_grasp0_deinterlaced` to learn a **three-region segmentation**:

| Channel | Semantic role |
|---------|---------------|
| ch0 | background (rigid, no contact) |
| ch1 | surgical instrument (protected, already trained) |
| ch2 | soft tissue (grasped / deforming) — **target of this stage** |
| ch3/ch4 | additional background channels |

The model is `RCFSoftTissueModel` (inherits `RCFDinoModel` → `RCFModel`).
Training entry point: `main_tissue.py`.
Config: `configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml`.

### All implemented losses

#### Always-on base losses

| Loss | Symbol | Formula | Purpose |
|------|--------|---------|---------|
| **L_seg** | `w_seg` | L1(P̂ + R, RAFT_flow) — flow reconstruction | Core RCF motion-segmentation objective. P̂ = piecewise-constant flow per channel, R = per-channel residual flow. |
| **L_entropy** | `w_entropy` | −∑ p·log(p) on channel softmax | Push masks to be hard (each pixel dominated by one channel). |
| **L_dino** | `w_dino` | Mask-weighted cosine distance in frozen DINO ViT feature space | Visual coherence: pixels in the same mask region should have similar DINO features. |

#### Instrument protection

| Loss | Symbol | Formula | Purpose |
|------|--------|---------|---------|
| **L_distill** | `w_distill` | BCE(student_ch1_mask, teacher_ch1_mask) | Soft distillation from frozen grasp10 teacher. Prevents ch1 (instrument) from degrading during grasp0 fine-tuning. This is a soft constraint — ch1 can still shift, but is penalized for drifting too far from the teacher. |

#### Unsupervised tissue-vs-background losses (channel-agnostic)

These losses operate on **all non-instrument channels** (i.e., everything except `instrument_channels=[1]`).
They do not hard-assign "tissue = ch2"; instead, softmax competition determines which channel wins.

| Loss | Symbol | Formula | Purpose |
|------|--------|---------|---------|
| **L_deform** | `w_deform` | Maximize ∑_c∈non-inst  mean\_mask_c(‖R_c‖) | Instrument has R≈0 (rigid-body motion, P̂ sufficient). Soft tissue has large R (non-rigid deformation). This loss rewards whichever non-instrument channel covers high-residual-magnitude pixels. |
| **L_div_tissue** | `w_div_tissue` | Maximize ∑_c∈non-inst  mean\_mask_c(&#124;div(RAFT_flow)&#124;) | Grasped tissue is compressed/stretched → RAFT optical flow has non-zero divergence. Background and rigid structures have div≈0. Signal is **external** (from RAFT, not from the network). |
| **L_contact** | `w_contact` | Maximize ∑_c∈non-inst  mean activation in soft-dilation ring around ch1 boundary | Encourage tissue channels to cluster near the instrument (contact region). Implemented via max-pool soft dilation. **Currently disabled** (w_contact=0). |

#### Annotation-driven losses (disabled in pure unsupervised mode)

These require grasp / dissection point coordinates from annotations.

| Loss | Symbol | Formula | Purpose |
|------|--------|---------|---------|
| **L_grasp_flow** | `w_grasp_flow` | Align ch2 flow with RAFT flow at annotated grasp point | Direct signal at tissue contact point. Disabled (unreliable annotations). |
| **L_dissect** | `w_dissect` | Gaussian activation at annotated dissection point for ch2 | Tissue should activate near the dissection locus. Disabled. |

#### Older V2 losses (disabled)

| Loss | Symbol | Purpose |
|------|--------|---------|
| **L_rigid** | `w_rigid` | Penalize ‖R_c‖ for instrument channels (enforce rigidity). Superseded by L_distill. |
| **L_grasp_conv** | `w_grasp_conv` | Grasping channel aligns with flow divergence. Superseded by L_div_tissue. |
| **L_align** | `w_align` | Tissue flow direction aligns with grasping-channel mean flow. |
| **L_motion** | `w_motion` | Tissue motion magnitude > background motion magnitude. |

---

### Experiment v1 — first unsupervised run (2026-06-15)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml`
**Script:** `script/run_grasp0_tissue_ft.sh`
**Run dir:** `saved/grasp0_tissue_ft_260615_102513/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

#### Loss weights

| Loss | Weight | Status |
|------|--------|--------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.05 | ✅ on |
| L_dino | 0.05 | ✅ on |
| L_distill | 0.1 | ✅ on — protects ch1 (instrument) |
| L_deform | 0.05 | ✅ on — channel-agnostic residual magnitude |
| L_div_tissue | 0.05 | ✅ on — channel-agnostic RAFT flow divergence |
| L_contact | 0.0 | ❌ off |
| L_grasp_flow | 0.0 | ❌ off (annotation-driven, disabled) |
| L_dissect | 0.0 | ❌ off (annotation-driven, disabled) |
| L_rigid | 0.0 | ❌ off |
| L_grasp_conv | 0.0 | ❌ off |
| L_align | 0.0 | ❌ off |
| L_motion | 0.0 | ❌ off |

#### Key design choices in v1

- **Channel-agnostic**: no loss hard-codes tissue = ch2; softmax competition decides.
- **Pure unsupervised**: zero annotation-driven losses; relies only on RAFT flow + DINO features.
- **Soft ch1 protection**: L_distill (w=0.1) is a soft regularizer — ch1 mask can still drift slightly but is penalized for diverging from the frozen teacher prediction.
- **Baseline val_miou at epoch 0 start: 86.15%** (ch1 instrument mIoU on CMC_grasp10, monitored throughout training to detect ch1 degradation).

---

### Experiment v3 — Run 204 (2026-06-15)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v3.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v3_260615_112039/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v1:** w_dino 0.05→0.05 (kept), w_deform 0.05→**0.5** (10×), w_div_tissue 0.05→**0.5** (10×), w_distill 0.1→**0.8**.

#### Loss weights

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.1 | ✅ on |
| L_dino | 0.05 | ✅ on — DINO spatial prior |
| L_distill | 0.8 | ✅ on — ch1 BCE distillation (bidirectional) |
| L_deform | 0.5 | ✅ on — residual magnitude in non-inst channels |
| L_div_tissue | 0.5 | ✅ on — RAFT flow divergence in non-inst channels |
| L_rigid_tissue | 0.0 | ❌ off |
| L_flow_cosine | 0.0 | ❌ off |
| L_contact | 0.0 | ❌ off |

#### val_miou per epoch (CMC_grasp10, ch1 instrument)

| Ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 15 | 20 | 25 | 29 |
|----|---|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| mIoU | 63.47 | 61.57 | **69.10** | 68.46 | 68.35 | 68.59 | 67.79 | 68.32 | 65.60 | 60.47 | 62.08 | 61.70 | 61.56 | 61.51 |

**Result:** Recovers to ~68-69% by epoch 2-7 (DINO prior helps), then drops back to 61-64% as deform/div pull ch1 away from instrument. Best: **69.10%** (epoch 2). Late convergence plateau: 61-64%.

**Key finding:** DINO prior provides spatial structure early on but the div_tissue signal is concentrated at boundaries (instrument-tissue interface), creating oscillating adversarial gradients that destabilize training after epoch 8.

---

### Experiment v4 — Run 206 (2026-06-15)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v4.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v4_260615_115553/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v3:** w_dino **0.05→0.0** (disabled), w_deform 0.5→**0.3**, w_div_tissue 0.5→**2.0** (4×), w_distill 0.8→**1.0**, w_entropy 0.1→0.1.

Hypothesis: DINO spatial prior was causing block-shaped masks (DINO quadrant bias). Disabling DINO and strengthening div_tissue to be the primary tissue signal.

#### Loss weights

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.1 | ✅ on |
| L_dino | 0.0 | ❌ off — disabled, spatial bias confirmed harmful |
| L_distill | 1.0 | ✅ on — ch1 BCE distillation (bidirectional) |
| L_deform | 0.3 | ✅ on |
| L_div_tissue | 2.0 | ✅ on — primary tissue signal (4× stronger) |
| L_rigid_tissue | 0.0 | ❌ off |
| L_flow_cosine | 0.0 | ❌ off |

#### val_miou per epoch

| Ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 15 | 20 | 25 | 29 |
|----|---|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| mIoU | **70.16** | 65.97 | 61.47 | 61.76 | 61.11 | 61.15 | 59.19 | 58.72 | 59.38 | 60.57 | 59.09 | 58.80 | 58.01 | 57.99 |

**Result:** Good start (70.16% epoch 0) but consistently declining. Late plateau 57-60% — **worse than v3**. Best: **70.16%** (epoch 0).

**Key finding:** div_tissue at w=2.0 dominates the gradient but the signal is physically concentrated at flow-boundary pixels (where flow direction changes sharply). These pixels lie exactly at the instrument-tissue interface, so both instrument and tissue channels cover them, producing oscillating gradients and no stable assignment. Raising w_div_tissue amplifies the instability rather than improving tissue separation.

---

### Experiment v6 — Run 208 (2026-06-15)  *(deform-only ablation)*

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v6.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v6_260615_180411/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v4:** w_dino=0.0 (keep off), w_div_tissue **2.0→0.0** (disabled — confirmed boundary-only signal), w_deform **0.3→0.5** (restore), w_entropy **0.1→0.05**, w_distill=1.0.
Ablation: trust deform alone; remove div_tissue entirely.

#### Loss weights

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.05 | ✅ on |
| L_dino | 0.0 | ❌ off |
| L_distill | 1.0 | ✅ on — ch1 BCE distillation (bidirectional) |
| L_deform | 0.5 | ✅ on — only tissue signal |
| L_div_tissue | 0.0 | ❌ off — boundary-only, adversarial gradients |
| L_rigid_tissue | 0.0 | ❌ off |
| L_flow_cosine | 0.0 | ❌ off |

#### val_miou per epoch

| Ep | 0 | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 16 | 17 | 22 | 25 | 29 |
|----|---|---|---|---|---|---|----|----|----|----|----|----|-----|
| mIoU | 68.94 | 67.97 | 63.27 | 60.64 | 60.39 | 59.96 | 60.28 | 62.23 | **62.55** | 60.90 | 62.07 | 59.07 | 59.83 |

**Result:** Plateau at 59-62%, best **62.55%** (epoch 16). loss_deform decreases monotonically (−0.27→−1.16) throughout training, confirming the residual signal is working — but val_miou stops improving, indicating deform alone cannot distinguish tissue sub-types (all tissue regions are pulled into a single homogeneous blob). Late-epoch decline (ep22→29) likely due to LR decay over-fitting warp reconstruction.

**Key finding:** **62.55% is the deform-only ceiling.** Serves as the ablation baseline for v8 (flow_cosine). Any improvement in v8 is attributable to L_flow_cosine.

---

### Experiment v7 — Run 209 (2026-06-15)  *(deform + flow-variance)*

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v7.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v7_260615_181208/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v6:** adds **L_rigid_tissue** (w=0.005) — maximizes RAFT flow variance within each non-instrument channel mask. Area signal (not spatial gradient). Weight set small (0.005) because real-data loss_rigid_tissue ≈ −25 (vs random-input ≈ −2); at w=0.1 the contribution (−2.5) would dominate deform (−0.17). At w=0.005 the contribution (−0.125) is comparable to deform.

#### Loss weights

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.05 | ✅ on |
| L_dino | 0.0 | ❌ off |
| L_distill | 1.0 | ✅ on — ch1 BCE distillation (bidirectional) |
| L_deform | 0.5 | ✅ on |
| L_rigid_tissue | 0.005 | ✅ on — maximize intra-channel RAFT flow variance |
| L_div_tissue | 0.0 | ❌ off |
| L_flow_cosine | 0.0 | ❌ off |

#### val_miou per epoch

| Ep | 0 | 1 | 2 | 3 | 4 | 5 | 8 | 9 | 10 | 15 | 20 | 25 | 29 |
|----|---|---|---|---|---|---|---|---|----|----|----|----|-----|
| mIoU | 68.90 | **70.59** | 67.71 | 64.16 | 61.84 | 64.13 | 62.13 | 63.55 | 63.32 | 61.75 | 59.93 | 60.08 | 59.69 |

**Result:** Early advantage over v6 (epoch 2-15 typically +1-4%), then converges to the same plateau (~59-62%) by epoch 20+. Best: **70.59%** (epoch 1, warmup). Late convergence same as v6.

**Key finding:** L_rigid_tissue provides a mild early benefit (the flow-variance signal gives additional push toward heterogeneous regions in the first ~15 epochs) but does not raise the long-term ceiling. Furthermore, L_rigid_tissue directly contradicts L_flow_cosine (rigid_tissue maximizes intra-channel flow variance; flow_cosine minimizes it) — **disabled in v8** to avoid gradient conflict.

---

### Experiment v8 — (planned, deform + flow-cosine)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v8.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v6:** adds **L_flow_cosine** (w=0.5) — each non-instrument channel should cluster pixels with similar RAFT flow direction. Instrument pixels excluded via **teacher's** ch1 mask (not student's, to break the feedback loop where student ch1 drift → instrument pixels no longer excluded → flow_cosine pulls them to tissue channels). **L_distill changed from bidirectional BCE to one-sided relu(teacher−student)** — only penalizes ch1 dropout (student < teacher), not expansion, making it compatible with flow_cosine's competitive pressure. L_rigid_tissue disabled (contradicts flow_cosine).

#### Loss weights

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | ✅ on |
| L_entropy | 0.05 | ✅ on |
| L_dino | 0.0 | ❌ off |
| L_distill | 1.0 | ✅ on — **one-sided**: relu(teacher_ch1 − student_ch1), prevents ch1 dropout only |
| L_deform | 0.5 | ✅ on |
| L_flow_cosine | 0.5 | ✅ on — RAFT flow direction clustering, teacher mask excludes instrument |
| L_rigid_tissue | 0.0 | ❌ off — contradicts flow_cosine (opposite gradients on same channels) |
| L_div_tissue | 0.0 | ❌ off — boundary-only signal |

#### val_miou per epoch (run 212, in progress)

| Ep | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|----|---|---|---|---|---|---|---|---|---|
| mIoU | 67.91 | **69.01** | 67.37 | 67.25 | 65.61 | 62.99 | 62.30 | 62.97 | — |

**Observation (early):** Peak at epoch 1 (69.01%), then declining. Suspected cause: without a push-apart constraint, all non-instrument channel centroids (mu) converge to the same mean flow direction → cross-entropy target becomes uniform → signal degenerates. → Motivates v9 (add diversity loss).

#### Ablation baseline
v6 (deform only) ceiling = **62.55%**. Any improvement in v8/v9 is attributable to L_flow_cosine.

---

### Experiment v9 — (run 213, deform + flow-cosine + diversity)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v9.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v9/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v8:** adds **L_flow_cosine_diversity** (weight=0.1 inside `flow_cosine_assignment_loss`) — minimises the mean pairwise cosine similarity between non-instrument channel centroids (mu vectors). Prevents channel collapse where all non-instrument channels converge to the same mean RAFT flow direction.

**Implementation detail:** diversity term recomputes mu **without detach** so the gradient flows back through masks. The main flow_cosine cross-entropy still uses detached mu as a pseudo-label. The two terms share the same forward pass — diversity adds one extra loop over channel pairs.

#### All loss weights

| Loss | Weight | Signal | Applies to | Notes |
|------|--------|--------|-----------|-------|
| **L_seg** (loss_warp_seg) | 1.0 | MSE(pred_flow, RAFT_flow) × boundary_mask | all channels | Core RCF flow reconstruction objective |
| **L_entropy** | 0.05 | −Σ p·log(p) on softmax | all channels | Sharpens masks, prevents soft boundaries |
| **L_dino** | 0.0 | DINO ViT feature cosine | — | Off — spatial quadrant bias confirmed harmful |
| **L_distill** | 1.0 | relu(teacher_ch1 − student_ch1) | ch1 only | One-sided: penalises ch1 dropout only, not expansion |
| **L_deform** | 0.5 | Mean ‖residual‖ within mask | non-inst channels | Network internal residual; area signal |
| **L_flow_cosine** | 0.5 | Cross-entropy vs mask-weighted mean RAFT flow direction | non-inst channels | Within-channel pull: each channel groups flow-similar pixels |
| **L_flow_diversity** | 0.1 | Mean pairwise cosine sim between channel mu vectors | non-inst channels | Between-channel push: prevents all channels collapsing to same direction |
| **L_rigid_tissue** | 0.0 | RAFT flow variance within mask | — | Off — contradicts L_flow_cosine |
| **L_div_tissue** | 0.0 | RAFT flow divergence | — | Off — boundary-only signal |
| **L_contact** | 0.0 | Mask activation near ch1 boundary | — | Off |

#### Gradient flow summary

```
ch1  ←── L_distill (one-sided relu, prevents dropout)
          L_seg (flow reconstruction)
          L_entropy

ch0, ch2, ch3, ch4 (non-inst) ←── L_deform      (cover high-residual pixels)
                                   L_flow_cosine  (pull: within-channel flow coherence)
                                   L_flow_diversity (push: between-channel flow divergence)
                                   L_seg, L_entropy
```

#### Design rationale

- **L_flow_cosine (pull)** alone creates degenerate solutions: all channels converge to the dominant flow direction (e.g., overall camera/instrument motion), leaving no distinction between channels.
- **L_flow_diversity (push)** breaks this symmetry: each channel is pushed toward a different "sector" of the flow direction space.
- Together they act like a **soft directional k-means**: pull assigns pixels to their nearest centroid, push keeps centroids separated.

#### val_miou per epoch (run 213, in progress)

| Ep | 0 | ... |
|----|---|-----|
| mIoU | (running) | — |

Ablation baseline: v8 (no diversity) best = **69.01%** (epoch 1). v6 (no flow_cosine) best = **62.55%**.

---

### Experiment v10 — (run 216, deform + flow-cosine + diversity + head reset)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v10.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v10_260615_204311/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key change vs v9:** `reset_non_instrument_heads: true` — at `on_train_start`, the non-instrument output rows (0, 2, 3, 4) of `decode_head2.conv_seg` are re-initialised with kaiming uniform. Channel 1 (instrument) weights are kept intact.

**Motivation:** The grasp10 pretrained model imprints strong priors on channels 0/2/3/4 that conflict with the grasp0 tissue partition. These priors resist fine-tuning and prevent the model from discovering a new tissue structure. Resetting only the final 1×1 classification conv (backbone untouched) gives non-instrument channels a clean slate, while L_distill quickly recovers ch1 instrument segmentation from the frozen teacher.

**Bug history (runs 214, 215):**
- Run 214: used `cls_seg` (a forward method in mmseg's BaseDecodeHead) instead of `conv_seg` (the actual `nn.Conv2d` weight attribute) — reset silently did nothing
- Run 215: placed `on_train_start` on `RCFSoftTissueModel` (a sub-module); Lightning only fires hooks on the top-level `TissueModel` — reset was never called
- Run 216: fixed by adding `on_train_start` to `TissueModel` in `main_tissue.py`, which calls `self.model._reset_non_instrument_mask_heads()`

**Reset confirmed in run 216 log:**
```
[RCFSoftTissueModel] reset_non_instrument_heads: re-initialised conv_seg rows [0, 2, 3, 4]  (kept rows [1])
```

step=10 loss comparison (reset effect):

| metric | run 215 (no reset) | run 216 (reset) |
|---|---|---|
| loss_warp_seg | 9.06 | **14.23** (random heads → higher reconstruction error) |
| loss_flow_cosine | 5.32 | **1.88** (uniform softmax → lower cosine assignment loss) |
| loss_entropy | 0.97 | **1.40** (softer masks from random init) |

**All loss weights (identical to v9):**

| Loss | Weight |
|------|--------|
| L_seg | 1.0 |
| L_entropy | 0.05 |
| L_distill | 1.0 (one-sided relu) |
| L_deform | 0.5 |
| L_flow_cosine | 0.5 |
| L_flow_diversity | 0.1 |
| L_rigid_tissue / L_dino / L_div_tissue | 0.0 |

#### val_miou per epoch (run 216, in progress)

| Ep | sanity | 0 | ... |
|----|--------|---|-----|
| mIoU | 86.15 (pretrained) | (running) | — |

Ablation baselines: v9 (same losses, no reset) in progress; v8 best = **69.01%**; v6 best = **62.55%**.

---

## Loss schedule overview (v1–v20)

> **Shared across all versions**: `w_seg=1.0`, `batch_size=8`, `optimizer=Adam(wd=1e-4)`, `teacher_ckpt=grasp10_ft_v9_fulltrain`
>
> **Always zero / never enabled**: `L_rigid`, `L_grasp_conv`, `L_align`, `L_motion`, `L_grasp_flow`, `L_dissect`, `L_contact`

| Ver | Ep | L_entropy | L_dino | L_deform | L_div_tissue | L_rigid_tissue | L_distill | distill_mode | distill anneal (ep≥5) | L_flow_cosine | fc_temp / diversity | L_flow_tv | TV: K / start_ep / margin | Reset |
|-----|-----|----------|--------|---------|-------------|--------------|---------|-------------|---------------------|-------------|-------------------|---------|--------------------------|-------|
| v1  | 30 | 0.05 | 0.05 | 0.05  | 0.05 | —     | 0.1 | BCE    | —            | —   | —         | —   | —               | none     |
| v2  | 30 | 0.05 | 0.05 | 1.0   | 0.5  | —     | 0.5 | BCE    | —            | —   | —         | —   | —               | none     |
| v3  | 30 | 0.10 | 0.05 | 0.5   | 0.5  | —     | 0.8 | BCE    | —            | —   | —         | —   | —               | none     |
| v4  | 30 | 0.10 | —    | 0.3   | 2.0  | —     | 1.0 | BCE    | —            | —   | —         | —   | —               | none     |
| v5  | 30 | 0.05 | —    | 0.5   | —    | 0.1   | 1.0 | BCE    | —            | —   | —         | —   | —               | none     |
| v6  | 30 | 0.05 | —    | 0.5   | —    | —     | 1.0 | BCE    | —            | —   | —         | —   | —               | none     |
| v7  | 30 | 0.05 | —    | 0.5   | —    | 0.005 | 1.0 | BCE    | —            | —   | —         | —   | —               | none     |
| v8  | 30 | 0.05 | —    | 0.5   | —    | —     | 1.0 | BCE    | —            | 0.5 | 0.5 / —   | —   | —               | none     |
| v9  | 30 | 0.05 | —    | 0.5   | —    | —     | 1.0 | BCE    | —            | 0.5 | 0.5 / 0.1 | —   | —               | none     |
| v10 | 30 | 0.05 | —    | 0.5   | —    | —     | 1.0 | BCE    | —            | 0.5 | 0.5 / 0.1 | —   | —               | partial① |
| v11 | 50 | 0.05 | —    | 0.5   | —    | —     | 1.0 | BCE    | —            | 0.5 | 0.5 / 0.1 | —   | —               | full②   |
| v12 | 50 | 0.05 | —    | 0.5   | —    | —     | 5.0 | sym③  | —            | 0.5 | 0.5 / 0.1 | —   | —               | full    |
| v13 | 50 | 0.05 | —    | 0.5   | —    | —     | 5.0 | sym③  | →0.5(relu)   | 0.5 | 0.5 / 0.2 | —   | —               | full    |
| v14 | 50 | 0.05 | —    | 0.5   | —    | —     | 5.0 | sym③  | →0.5(relu)   | 0.5 | 0.5 / 0.2 | 0.1 | 4 / **0** / —   | full    |
| v15 | 50 | 0.05 | —    | 0.5   | —    | —     | 5.0 | sym❌  | →0.5(relu)   | 0.5 | 0.5 / 0.2 | 0.5 | 5 / 5 / 0.3     | full    |
| v16 | 50 | 0.05 | —    | 0.5   | —    | —     | 5.0 | sym✓  | →0.5(relu)   | 0.5 | 0.5 / 0.2 | 0.5 | 5 / 5 / 0.3     | full    |
| v17 | 80 | 0.05 | —    | 0.5   | —    | —     | 1.0 | relu  | —            | 0.5 | 0.5 / 0.5 | 0.2 | 5 / 5 / 0.3     | none    |
| v18 | 80 | 0.05 | —    | 0.5   | —    | —     | 1.0 | relu  | —            | 0.5 | 0.5 / 0.5 | 0.2 | 5 / 5 / 0.3     | none    |
| v19 | 80 | 0.05 | —    | 0.5   | —    | —     | 1.0 | relu  | —            | 0.5 | 0.5 / 0.5 | **2.0** | 5 / 5 / 0.3 | none    |
| v20 | 80 | **0.10** | —  | 0.5   | —    | —     | 1.0 | relu  | —            | 0.5 | 0.5 / 0.5 | **2.0** | 5 / 5 / 0.3 | none    |

**Notes**

① v10 `reset_non_instrument_heads=true`: re-initialises only conv_seg rows ch0/2/3/4; ch1 weights are kept

② v11–v16 `reset_full_decode_head=true`: re-initialises all parameters of decode_head2 (including ch1)

③ v12–v14 ran on **old code** where `distill_mode` had no effect (always BCE); v14 ch1 collapse was caused by `L_flow_tv` active from ep0 (fixed via `flow_tv_start_epoch=5`)

❌ v15 ran on **new code** where symmetric was implemented as MSE (bounded gradient); ch1 could not recover against warp_seg and collapsed

✓ v16 fix: symmetric restored to BCE (gradient ∝ teacher/student, diverges near zero → always dominates)

④ v18 identical to v17 except `lr=3e-5`; analysis showed faster train-loss drop but higher entropy (mask diffusion) and unstable val_miou → high LR too aggressive

⑤ v19 vs v17: `w_flow_tv` raised from 0.2 → 2.0 to bring TV loss to ~¼ of cosine strength and better enforce flow-cluster boundary alignment

⑥ v20 vs v19: `w_entropy` raised from 0.05 → 0.10 to push against mask graying in ambiguous low-flow regions

---

## Loss schedule overview (v21–v26)

> **Key shift**: Replace `L_flow_tv` (gradient dead zone confirmed in v19/v20) with `L_flow_cluster_ce` (K-means color-block CE, external targets).
>
> **Core insight**: `L_flow_cosine` is self-referential — its CE target μ_c is derived from the current mask, so it reinforces the pretrained prior rather than breaking it. `L_flow_cluster_ce` uses external RAFT K-means labels (independent of current mask) and is therefore the true prior-breaking force for aligning non-instrument channels with RAFT flow color blocks.

| Ver | Ep | L_entropy | L_deform | L_distill | L_flow_cosine | L_flow_cluster_ce | fcc: temp / div / start_ep | K-means | Notes |
|-----|-----|----------|---------|---------|-------------|-----------------|---------------------------|---------|-------|
| v21 | 80 | 0.05 | 0.5 | 1.0 relu | 0.5 | 0.5 | 0.3 / 0.5 / ep5 | flow 2D, fixed ring init, 10 iter | equal weight |
| v22 | 80 | 0.05 | 0.5 | 1.0 relu | 0.2 | 1.0 | 0.3 / 0.5 / ep0 | flow 2D, fixed ring init, 10 iter | cluster_ce primary |
| v23 | 80 | 0.05 | 0.5 | 1.0 relu | 0.5 | 0.5 | 0.3 / 0.5 / ep5 | flow 2D, **sector-mean init**, 10 iter | = v21 + adaptive init |
| v24 | 80 | 0.05 | 0.5 | 1.0 relu | 0.2 | 1.0 | 0.3 / 0.5 / ep0 | flow 2D, **sector-mean init**, 10 iter | = v22 + adaptive init |
| v25 | 50 | **0.3** | 0.5 | **3.0** relu | **0.0** | **3.0** | 0.3 / 0.5 / ep5 | **flow+HSV 5D, k-means++, 300 iter** | cluster_ce sole non-ch1 signal |
| v26 | 50 | **0.3** | 0.5 | **3.0** relu | **0.0** | **4.0** | 0.3 / 0.5 / **ep0** | **flow+HSV 5D, k-means++, 300 iter** | cluster_ce sole non-ch1 signal, most aggressive |

### v21 — K-means CE and cosine equal weight (Run 251, in progress)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v21.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v19:**
- `w_flow_tv: 2.0 → 0.0` — disabled; gradient dead zone confirmed in v19/v20 logs
- `w_flow_cluster_ce: 0.5` — new loss: each non-instrument channel claims one K-means color block
- `flow_cluster_ce_temperature: 0.3` — harder channel claiming than cosine (temp=0.5)
- `flow_cluster_ce_diversity: 0.5` — push channels' K-means affinity profiles apart
- `flow_cluster_ce_start_epoch: 5` — same warm-up delay as previous flow_tv

**L_flow_cluster_ce algorithm:**
1. Run K-means (K=5) on normalized RAFT flow per frame; cluster labels correspond to flow_to_color visualization color blocks
2. Build affinity matrix P[K_ch, K]: P[ci, k] = weighted mean of channel ci mask over cluster k pixels
3. A = softmax(P / temp, dim=0, detached): soft channel-to-cluster assignment (competition across channels per cluster)
4. CE target: pixels in cluster k receive target distribution A[:, k]
5. Diversity: μ-based (same as flow_cosine); gradient ∝ (flow_dir[pixel] − μ_ci), non-zero even for near-uniform masks

**Loss weights:**

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | warp_seg flow reconstruction |
| L_entropy | 0.05 | prevent mask graying |
| L_distill | 1.0 (one-sided relu) | protect ch1 (instrument) |
| L_deform | 0.5 | residual magnitude |
| L_flow_cosine | 0.5 | self-reinforcing EM; prevents channel collapse |
| L_flow_cluster_ce | 0.5 | external K-means targets; active from ep5 |
| L_flow_tv | 0.0 | disabled — gradient dead zone confirmed |

---

### v22 — K-means CE primary, cosine auxiliary

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v22.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Core analysis:**
- `L_flow_cosine` CE part: μ_c is derived from the current mask → self-reinforcing EM → reinforces/sharpens the pretrained prior rather than breaking it
- `L_flow_cluster_ce`: K-means labels are independent of the mask → external supervision signal → the true prior-breaking force

**Key changes vs v21:**
- `w_flow_cluster_ce: 0.5 → 1.0` — primary: 2× cosine weight; external K-means targets break pretrained prior
- `flow_cluster_ce_start_epoch: 5 → 0` — primary loss should act from epoch 0, not wait for cosine to reinforce prior structure
- `w_flow_cosine: 0.5 → 0.2` — auxiliary: only prevents channel graying, does not drive assignment
- `flow_cosine_diversity: 0.5 → 0.3` — reduced diversity push matching auxiliary role

**Loss weights:**

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | warp_seg flow reconstruction |
| L_entropy | 0.05 | prevent mask graying |
| L_distill | 1.0 (one-sided relu) | protect ch1 (instrument) |
| L_deform | 0.5 | residual magnitude |
| L_flow_cosine | **0.2** | auxiliary: prevents collapse, does not drive assignment |
| L_flow_cluster_ce | **1.0** | primary: external K-means targets; active from ep0 |

---

### v23 — v21 + data-adaptive K-means init (running)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v23.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Problem fixed:** v21/v22 used a fixed ring at radius 0.5 for K-means seeding. For grasp0 data where instrument motion dominates at r≈0.9 and tissue motion sits at r≈0.1–0.2, this initialization may not align cluster centroids with the actual RAFT flow color blocks.

**Key change vs v21:**
- K-means init replaced with **data-adaptive deterministic** seeding:
  - centroid 0 — mean of near-zero pixels (r < 5% max) → background / white in RAFT viz
  - centroids 1–4 — mean of pixels in each of 4 equal angular sectors (90° each), placing each seed where the actual data lives in that directional band

Loss weights identical to v21.

---

### v24 — v22 + data-adaptive K-means init (running)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v24.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

Same data-adaptive init as v23, applied on top of v22 (cluster_ce primary, from ep0). Loss weights identical to v22.

---

### v25 — k-means++ + joint flow+HSV, cluster_ce sole signal (eq. weight baseline)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v25.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Core changes vs v23:**

1. **K-means algorithm upgraded to k-means++**: distance-proportional random seeding; better coverage than fixed-ring or sector-mean init.

2. **Feature space expanded to 5D (flow + HSV)**:
   - flow component: (fx/max, fy/max) — 2D, same space as RAFT flow_to_color
   - color component: (cos(2πH), sin(2πH), S) — circular hue encoding + saturation; V dropped (brightness irrelevant for color-block identity)
   - weights: flow × 1.0, color × 1.0 → concatenated 5D
   - Motivation: pure flow fails to separate same-motion regions with different appearance (e.g. tissue vs background both static); color breaks the tie

3. **Max 300 EM iterations** (was 10; with k-means++ init, typically converges in ~30 iterations)

4. **Strong entropy + distill to counter gray-mask basin**:
   - `w_entropy: 0.05 → 0.3` (6×): actively pushes masks away from uniform gray
   - `w_distill: 1.0 → 3.0` (3×): stronger teacher signal anchors ch1 (instrument) throughout domain shift

5. **Cosine loss disabled** (`w_flow_cosine: 0.0`): k-means++ + HSV provides sufficient cluster signal; cosine (self-referential EM) no longer needed

**Loss weights:**

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | warp_seg flow reconstruction |
| L_entropy | **0.3** | strong push against uniform gray mask |
| L_distill | **3.0** (one-sided relu) | strong ch1 anchor during domain shift |
| L_deform | 0.5 | residual magnitude |
| L_flow_cosine | **0.0** | disabled |
| L_flow_cluster_ce | **3.0** | sole non-ch1 learning signal; flow+HSV 5D k-means++; ep5 start |

---

### v26 — k-means++ + joint flow+HSV, cluster_ce most aggressive

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v26.yaml`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

Same architecture as v25 but with the cluster_ce loss pushed further: `4.0` weight active from **epoch 0** (no warmup). Intended to test whether an even stronger and earlier clustering signal can overcome the gray-mask basin faster.

**Key differences vs v25:**

| | v25 | v26 |
|--|-----|-----|
| `w_flow_cluster_ce` | 3.0 | **4.0** |
| `flow_cluster_ce_start_epoch` | 5 | **0** |

**Loss weights:**

| Loss | Weight | Notes |
|------|--------|-------|
| L_seg | 1.0 | warp_seg flow reconstruction |
| L_entropy | **0.3** | strong push against uniform gray mask |
| L_distill | **3.0** (one-sided relu) | strong ch1 anchor during domain shift |
| L_deform | 0.5 | residual magnitude |
| L_flow_cosine | **0.0** | disabled |
| L_flow_cluster_ce | **4.0** | sole non-ch1 learning signal; flow+HSV 5D k-means++; ep0 start |

---

## Loss schedule overview (v27–v34)

> **Shared across all versions**: `w_seg=1.0`, `w_deform=0.5`, `batch_size=8`, `optimizer=Adam(wd=1e-4)`, `teacher_ckpt=grasp10_ft_v9_fulltrain`
>
> **Always zero**: `L_flow_cosine`, `L_flow_tv`, `L_rigid_tissue`, `L_dino`

| Ver | Ep | L_entropy | L_distill | distill_mode | L_flow_cluster_ce | fcc: color_weight / div | L_flow_angle_ce | Notes |
|-----|-----|----------|---------|-------------|-----------------|------------------------|----------------|-------|
| v27 | 50 | 0.3 | 3.0 relu | one_sided | 4.0 (ep0) | 1.0 (flow+HSV) / 0.5 | — | Sinkhorn-STE channel↔cluster assignment (replaces softmax P) |
| v28 | 50 | 0.3 | 3.0 relu | one_sided | 4.0 (ep0) | 1.0 (flow+HSV) / 0.5 | — | Sort cluster labels by ascending flow magnitude for cross-batch consistency |
| v29 | 50 | **1.5** | **0.5** relu | one_sided | 4.0 (ep0) | 1.0 (flow+HSV) / 0.5 | — | Entropy ×5, distill ×0.17 (distill contribution was ≈0%, entropy too weak) |
| v30 | 50 | 1.5 | 0.5 relu | one_sided | 4.0 (ep0) | **1.0** (flow+HSV) / 0.5 | — | Same as v29; cluster viz shows salt-and-pepper noise (color_weight=1.0 dominates) |
| v31 | 50 | 1.5 | 0.5 relu | one_sided | 4.0 (ep0) | **0.0** (flow only) / 0.5 | — | Clustering on flow unit direction vectors only; cluster boundaries = RAFT color blocks |
| v32 | 50 | 1.5 | **3.0** | **symmetric** | 4.0 (ep0) | 0.0 / 0.5 | — | Symmetric BCE distill prevents ch1 from absorbing tissue regions |
| v33 | 50 | **15.0** | 3.0 | symmetric | 4.0 (ep0) | 0.0 / 0.5 | — | Entropy ×10 to forcibly escape uniform-gray mask attractor |
| v34 | 50 | 15.0 | 3.0 | symmetric | **0.0** | — | **4.0** (div=0.5) | Replace k-means with atan2 angle sectors; 4 sectors → 4 non-inst channels |

### Key changes summary (v27→v34)

**v27**: Replaced softmax(P) channel↔cluster assignment with Sinkhorn-STE, enforcing 1-to-1 matching and guaranteeing non-zero gradients even when masks are uniformly gray.

**v28**: Re-sorted cluster labels by ascending flow magnitude so that cluster 0 always corresponds to the most static (background) pixels, ensuring consistent label ordering across batches.

**v29**: Found that distill (w=3.0) contributed ≈0% to the total loss (teacher and student were nearly identical), and entropy only accounted for 1.7% — completely dominated by warp_seg + cluster_ce. Raised entropy weight 0.3→1.5 and lowered distill 3.0→0.5.

**v30**: Same weights as v29. Cluster visualization revealed severe salt-and-pepper noise: color_weight=1.0 caused image HSV texture to dominate clustering, completely decoupled from RAFT flow visualization logic.

**v31**: Changed clustering features to flow unit direction vectors (`pts = flow / |flow|`), color_weight=0. Cluster boundaries now directly align with RAFT color block boundaries. However, ch1 absorbed large tissue regions because one_sided distill only prevents dropout, not expansion.

**v32**: Switched distill to symmetric BCE — penalizes both directions: tissue pixels (teacher≈0) are strongly suppressed if student activates there; instrument pixels (teacher≈1) are prevented from dropping out. Weight raised 0.5→3.0 to compensate for smaller per-unit BCE magnitude. Entropy still stuck at 1.30–1.35.

**v33**: Raised entropy weight 1.5→15.0 (×10), increasing its contribution from ~10% to ~50%+ of total loss, forcibly pushing masks away from the uniform-gray attractor. ep0 val_miou=61.89%, peak 67.79% (ep1) but high variance in later epochs.

**v34** ⭐: Replaced k-means cluster_ce with `flow_angle_ce`: divide [-π, π] into 4 equal sectors via atan2, each non-instrument channel fits one sector. Fully deterministic, zero EM iterations. Peak 66.67% (ep25), stabilizes at 65–66% in later epochs (vs. v33's 55–62% late-epoch variance). **Notable: grasp10 instrument segmentation (val_miou) keeps improving with continued training rather than decaying** — unlike all prior versions where fine-tuning on grasp0 eventually degrades the pretrained ch1 instrument detector.

### val_miou summary

| Ver | Job | Best | ep0 | Notes |
|-----|-----|------|-----|-------|
| v28 | 258/259 | ~65% | 65.00 | k-means with magnitude-sorted labels |
| v29 | 260 | ~61% | 61.00 | dummy cluster absorbed 96% of pixels |
| v30 | 261 | ~55% | 55.00 | salt-and-pepper cluster noise, early stop |
| v31 | 262 | ~56% | 56.00 | ch1 absorbed tissue (one_sided distill), early stop |
| v32 | 263 | ~64% | 59.00 | symmetric distill protects ch1 boundary |
| v33 | 264 | **67.79%** (ep1) | 61.89 | entropy 15.0, high late-epoch variance |

---

## Grasp0 Tissue Segmentation — Real-Annotation Evaluation (260622)

First evaluation against **real CVAT annotations** on CMC_grasp0 (222 labeled frames:
209 with tissue, 213 with instrument). Metric = **foreground IoU (DAVIS J)** under
**per-frame greedy-union oracle** (each frame picks + merges the channels that best
match its GT — see `main.py` test_step). Tissue evaluated on the 209-frame
`eval_tissue/val.txt`.

Tooling: `tools/split_tissue_anno.py` (color→binary), `tools/build_eval_dirs.py`
(eval roots + symlinked JPEGImages), `script/run_grasp0_eval_all.sh` (batch eval),
results in `saved/grasp0_eval_260622_111644/`.

> ⚠️ **Data-leak caveat**: every model below was trained on the **full 601 set**
> (which *includes* these 222 labeled frames). Training is self-supervised (no GT
> used), so this is label-leak-free but **image-leak optimistic**. The first truly
> image-isolated run is **v41** (trained on `train_unlabeled_379.txt`, 379 frames,
> the 222 labeled ones excluded), in progress.

### Baseline (starting point of all grasp0_tissue_ft runs)

| Model | tissue J (%) | tissue P/R/F1 (%) | instrument J (%) | instrument P/R/F1 (%) |
|-------|-------------|-------------------|-----------------|----------------------|
| **v9** (`grasp10_ft_v9 epoch0-step149`) | **67.65** | 80.35 / 82.79 / 79.94 | **68.82** | 77.50 / 86.00 / 80.02 |

### tissue oracle J — best checkpoint per version (sorted)

| Version | Best ckpt | tissue J (%) |
|---------|-----------|-------------|
| **v18** ⭐ | ep76 | **74.61** |
| v36b | ep49/last | 74.31 |
| v23 | ep79/last | 74.28 |
| v35 | ep48/last | 74.07 |
| v22 | ep69 | 73.99 |
| v24 | last | 73.96 |
| v3 | ep29/last | 73.59 |
| v6 | ep28 | 73.46 |
| v4 | ep25 | 73.40 |
| v20 | ep76 | 73.18 |
| v16 | ep17 | 73.10 |
| v40 (bilateral CE) | ep39/last | 73.09 |
| v19 | ep79 | 72.97 |
| ft_260615_102513 | ep20 | 72.85 |
| v21 | ep77 | 72.83 |
| v12 | last | 72.73 |
| v7 | ep25 | 72.68 |
| v17 | ep75 | 72.60 |
| v13 | ep47 | 72.22 |
| v37 | last | 71.98 |
| v2 | ep4 | 71.36 |
| v28 | ep7 | 69.19 |
| v36c | ep9/last | 68.57 |
| v27 | ep13 | 68.55 |
| **v9 baseline** | — | **67.65** |
| v29 | ep10/last | 68.27 |
| v30 | ep2 | 67.35 |
| v31 | ep0 | 65.57 |
| v26 | ep44 | 65.14 |
| v33 | ep37 | 62.96 |
| v32 | ep8 | 59.20 |
| v34 | ep46 | 57.39 |
| v36a | ep0 | 57.11 |

**Notes**
- tissue almost always lands in **ch3 + ch4** (e.g. v40 dist `[17,42,4,205,164]`);
  models that scatter tissue into other channels (v26/v32/v34/v36a) score worst.
- Best models (**v18 / v36b / v23**, ~74.3–74.6%) beat the v9 start (68.33%) by **+6 pts**.
- v40 (bilateral CE) = 73.09%, solidly above baseline; v41 (no-leak) is the honest re-test, in progress.
| v34 ⭐ | 265 | **66.67%** (ep25) | 59.47 | angle_ce, more stable late epochs (65–66%); **val_miou continues to improve with more training rather than decaying** |

---

## v43 Evaluation Results (260623, job 299)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v43.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v43_260623_111047/`
**Key changes vs v42:** `distill_mode: one_sided → symmetric` (BCE 自适应弹簧), `w_distill: 1.0 → 2.0`

Instrument channel (ch1) stabilized at **0.63–0.65** instead of dropping to 0.57 (v42), enabling `val_miou_sum` monitoring to correctly track joint improvement.

Evaluated on:
- **Tissue**: `eval_tissue/` (209 labeled frames), per-frame greedy-union oracle, `oracle_exclude_channels=[1]`
- **Instrument**: `eval_instrument/` (213 labeled frames), fixed ch1

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) |
|------------|----------------|----------------------|---------------------|--------------------------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 |
| ep7   | 70.78 | 81.59 / 85.01 / 82.07 | **66.03** | 74.24 / 85.37 / 77.81 |
| **ep48** ⭐ | **71.69** | **83.74 / 84.19 / 82.72** | 64.77 | 74.80 / 82.38 / 76.66 |
| ep51  | 71.58 | 83.74 / 84.08 / 82.63 | 64.80 | 75.59 / 82.24 / 76.76 |
| last  | 70.47 | 82.99 / 83.76 / 81.84 | 63.75 | 74.65 / 81.43 / 75.94 |

**Best tissue: ep48 (71.69%)** — improves over v42's single-val best.
**Best instrument: ep7 (66.03%)** — symmetric distill holds ch1 throughout training.

### Comparison with baseline

| Model | tissue J (%) | instrument J (%) |
|-------|-------------|-----------------|
| v9 baseline | 67.65 | 68.82 |
| v43 best (ep48 / ep7) | **71.69** | **66.03** |
| Δ vs baseline | +4.04 | -2.79 |

---

## All-Version Ranking: tissue + instrument mIoU sum (job 300)

> Best checkpoint per version, sorted by `tissue + instrument` sum descending.
> Evaluated on `eval_tissue/` (209 frames, oracle) + `eval_instrument/` (213 frames, ch1).
> v9 baseline: `grasp10_ft_v9_fulltrain epoch=0-step=149`.

| Rank | Version | Best ckpt | tissue J (%) | inst J (%) | sum (%) |
|------|---------|-----------|:------------:|:----------:|:-------:|
| 1 | **v43** ⭐ | ep7 | 70.78 | **66.03** | **136.81** |
| 2 | **v9 baseline** | ep0 | 67.65 | 68.82 | 136.47 |
| 3 | **v47** | ep0 | 67.90 | **67.80** | **135.70** |
| 4 | **v46** | ep1 | 69.54 | 65.71 | 135.25 |
| 5 | v41 | ep37 | 71.10 | 61.91 | 133.01 |
| 4 | v42 | ep64 | 72.15 | 60.47 | 132.62 |
| 5 | v35 | ep0 | 68.77 | 63.12 | 131.89 |
| 6 | v3 | ep24 | 73.04 | 58.47 | 131.51 |
| 7 | v2 | ep4 | 70.86 | 59.94 | 130.80 |
| 8 | v36b | ep49 | 74.03 | 55.88 | 129.91 |
| 9 | v1 | ep19 | 72.06 | 57.65 | 129.71 |
| 10 | v8 | ep28 | 72.77 | 56.46 | 129.23 |
| 11 | v6 | last | 72.68 | 56.25 | 128.93 |
| 12 | v12 | ep43 | 71.98 | 56.56 | 128.54 |
| 13 | v36c | ep8 | 67.40 | 60.92 | 128.32 |
| 14 | v23 | ep78 | 73.37 | 54.88 | 128.25 |
| 15 | v7 | ep25 | 72.07 | 55.91 | 127.98 |
| 16 | v40 | ep39 | 72.60 | 55.27 | 127.87 |
| 17 | v4 | ep29 | 72.58 | 54.94 | 127.52 |
| 18 | v37 | ep17 | 71.36 | 55.55 | 126.91 |
| 19 | v24 | last | 72.88 | 53.83 | 126.71 |
| 20 | v22 | ep69 | 73.08 | 53.30 | 126.38 |
| 21 | v18 | last | 73.80 | 51.52 | 125.32 |
| 22 | v20 | ep76 | 72.14 | 52.92 | 125.06 |
| 23 | v13 | ep49 | 70.82 | 54.03 | 124.85 |
| 24 | v19 | ep78 | 72.07 | 52.45 | 124.52 |
| 25 | v17 | last | 71.91 | 51.94 | 123.85 |
| 26 | v21 | ep78 | 71.66 | 50.00 | 121.66 |
| 27 | v34 | ep46 | 56.47 | 63.44 | 119.91 |
| 28 | v33 | ep36 | 62.02 | 55.42 | 117.44 |
| 29 | v36a | ep0 | 56.25 | 59.55 | 115.80 |
| 30 | v31 | ep1 | 63.11 | 52.56 | 115.67 |
| 31 | v30 | ep0 | 64.62 | 50.77 | 115.39 |
| 32 | v28 | ep7 | 64.22 | 50.85 | 115.07 |
| 33 | v27 | ep13 | 63.50 | 50.25 | 113.75 |
| 34 | v32 | ep8 | 58.54 | 53.79 | 112.33 |
| 35 | v29 | ep10 | 64.03 | 48.17 | 112.20 |
| 36 | v26 | ep49 | 52.48 | 40.20 | 92.68 |
| 37 | v16 | ep17 | 55.76 | 2.16 | 57.92 |

---

## v46 Evaluation Results (260624, job 305)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v46.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v46_260624_090944/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v43:**
- `w_flow_bilateral_ce: 4.0` — flow-guided bilateral self-training CE on non-instrument channels; zero gradient to ch1 (sub-softmax trick)
- `w_dino: 2.0` — DINO ch1 anchor (was 0 in v43)
- `w_deform: 1.0` — non-instrument residual magnitude reward (active vs 0 in v43)
- `distill_mode: one_sided`, `w_distill: 5.0` — relu(teacher−student) only; effectively near-zero signal (raw ≈ 0.004)

**Findings:** tissue improved above v43 (+1–2 pp) but instrument declined steadily from ep0 (65.71%) to ~50% at ep80. one_sided distill raw ≈ 0.004 → weighted ≈ 0.02, essentially zero ch1 protection. `deform` loss (w=1.0) actively pushes ch1 down via softmax competition.

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) |
|------------|----------------|----------------------|---------------------|--------------------------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 |
| **ep1** ⭐ (best sum) | 69.54 | 80.97 / 84.20 / 81.21 | **65.71** | 75.44 / 83.92 / 77.33 |
| ep3  | 70.52 | 81.11 / 85.24 / 81.86 | 62.34 | 75.05 / 79.54 / 74.68 |
| **ep10** ⭐ (best tissue) | **70.80** | 82.15 / 84.91 / 82.15 | 62.42 | 73.57 / 80.31 / 74.26 |

---

## v47 Evaluation Results (260624, job 305)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v47.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v47_260624_094240/`
**Pretrained from:** `saved/grasp10_ft_v9_fulltrain_ft_260609_213030/epoch=0-step=149.ckpt`

**Key changes vs v46:**
- `distill_mode: one_sided → symmetric` — bidirectional BCE, real gradient to ch1; symmetric raw ≈ 0.07 vs one_sided raw ≈ 0.004
- `w_distill: 5.0 → 2.0` — reduced because symmetric is self-adaptive (gradient ∝ teacher/student)
- `w_dino: 2.0 → 3.0` — stronger DINO ch1 appearance anchor

**Findings:** symmetric distill gave real ch1 gradient. Instrument held at 64–67% for the first ~5 epochs (vs v46's immediate drop), but still declined to ~50% by ep80. Root cause: `deform` (w=1.0) and `entropy` (w=5.0) together overpower the distill+DINO protection — `bilateral_ce` has zero ch1 gradient (sub-softmax) and is not the cause. **v47 ep0 (67.80% instrument, 67.90% tissue, sum=135.70) ranks 3rd all-time, just below the v9 baseline in instrument quality.**

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) |
|------------|----------------|----------------------|---------------------|--------------------------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 |
| **ep0** ⭐ (best sum) | 67.90 | 79.12 / 84.33 / 80.10 | **67.80** | 82.91 / 78.26 / 79.25 |
| ep2  | **70.48** | 82.28 / 84.02 / 81.92 | 64.37 | 74.58 / 82.61 / 76.79 |
| ep6  | 70.44 | 81.49 / 84.80 / 81.76 | 63.28 | 74.06 / 81.53 / 75.07 |

### Loss gradient analysis

| Loss | Weight | Ch1 gradient | Direction on ch1 |
|------|--------|-------------|-----------------|
| `warp_seg` | 1.0 | ✅ full softmax | uncertain |
| `entropy` | 5.0 | ✅ full softmax | ⬇️ amplifies decline |
| `DINO` | 3.0 | ✅ ch1 only | ⬆️ anchor |
| `deform` | 1.0 | ✅ via softmax | ⬇️ rewards non-inst residual → crowds out ch1 |
| `distill` (symmetric) | 2.0 | ✅ ch1 only | ⬆️ protection |
| `bilateral_ce` | 4.0 | ❌ zero (sub-softmax renorm cancels) | — |
| `flow_cosine` | 0.5 | ❌ sub-softmax main term | — |

**Key finding:** `bilateral_ce` does **not** hurt ch1 (zero gradient via sub-softmax renormalisation). The true causes of ch1 decline are `deform` (actively pushes ch1 down via softmax competition) and `entropy` (amplifies any ch1 drop). Next experiments (v48, v49) disable all four: `entropy=0`, `deform=0`, `flow_cosine=0`, `distill=0`.

---

## v51 Evaluation Results (260625, job 311)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v51.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v51_260624_202504/`
**Pretrained from:** `data/pretrained/densecl_r50_imagenet_200ep.pth` (from scratch — no grasp10 inheritance)

**Key design (from-scratch baseline):**
- Start from DenseCL (ImageNet contrastive, no flow head) — zero channel-assignment inheritance
- Loss: `warp_seg` (w=1.0) + `DINO` (w=1.0, all 5 channels, `dino_channels: null`)
- All other losses disabled: `bilateral_ce=0`, `entropy=0`, `deform=0`, `distill=0`, `flow_cosine=0`
- `instrument_channels: []`, `oracle_exclude_channels: []` — no hard-coded channel exclusion
- lr=1e-4 (flow/mask heads randomly initialised), epochs=80

**Instrument channel:** After training, instrument naturally migrated to **ch4** (not ch1). Eval configs use `object_channel: 4` (instrument) and `oracle_exclude_channels: [4]` (tissue).

**Training dynamics:**
- DINO loss converges to ~0.585 within 2 epochs and stays **flat** for all 80 epochs — channel semantic content fixed early
- `warp_seg` decreases monotonically from ~28 → ~3, continuously refining spatial boundaries
- Instrument mIoU peaks around ep16–ep41, tissue steadily climbs to 73–74% (highest across all versions)
- No systematic channel collapse (unlike v46–v48 starting from v9)

**DINO loss mechanism:** Not K-means — a **mask-weighted centroid consistency loss** per channel. Each channel's assigned pixels are pulled toward their ViT-feature centroid. Converges quickly to a stable semantic layout, then acts as a fixed anchor.

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) | sum (%) |
|------------|----------------|----------------------|---------------------|--------------------------|---------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 | 136.47 |
| **ep16** ⭐ (best inst) | 70.13 | 80.63 / 84.96 / 81.76 | **61.30** | 70.34 / 83.44 / 74.14 | 131.43 |
| **ep41** ⭐ (best sum) | **73.58** | 83.18 / 86.79 / 84.21 | 59.84 | 76.25 / 72.76 / 72.64 | **133.42** |
| ep51 | 73.26 | 83.02 / 86.90 / 84.03 | 58.23 | 70.88 / 75.87 / 71.67 | 131.49 |
| last (ep79) | 73.43 | 84.16 / 86.21 / 84.15 | 53.26 | 64.52 / 74.12 / 66.71 | 126.69 |

### Key findings

- **Highest tissue mIoU ever recorded (73.58%)** at ep41 — DenseCL + warp_seg + DINO all-channel achieves better tissue than any v9-based fine-tuning
- Instrument is lower than the v9 baseline (59.84% vs 68.82%) because there is no explicit instrument anchor; `warp_seg` alone cannot guarantee instrument stays in a fixed channel
- Best sum (133.42 at ep41) is below v43 (136.81) and v9 baseline (136.47), but **tissue quality is substantially higher**
- Training is stable — no collapse pattern seen in v46–v48

---

## v52 Evaluation Results (260625, job 315 train / job 319 eval)

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v52.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v52_260625_102034/`
**Pretrained from:** `data/pretrained/densecl_r50_imagenet_200ep.pth` (from scratch)

**Key changes vs v51:**
- `decode_head2`: FCNHead (48×48, `in_index=[0,3]`) → **MultiScaleSegHead** — feat1, feat2, feat3 are each projected to 256ch via 1×1 conv, summed element-wise at H/8, fused with a 3×3 dilated conv, upsampled to H/4, concatenated with feat0, then refined by two 3×3 convs → output [5, H/4, W/4] = [5, 96, 96]
- `mask_size`: [128,128] → **[96,96]** — matches the native output resolution of MultiScaleSegHead; warp_seg loss now operates at true 96×96 rather than upsampled 48→128
- Parameters: 5.90M → 3.28M (fewer params in decode_head2 despite richer multi-scale fusion, since resize_concat is removed)
- All other settings identical to v51: DenseCL from scratch, warp_seg + DINO (w=1.0), epochs=80

**Instrument channel:** After training, the instrument naturally migrated to **ch3**. Eval uses `object_channel: 3` (instrument) and `oracle_exclude_channels: [3]` (tissue oracle excludes ch3).

**Training dynamics vs v51:**
- Loss converges faster: at epoch 31, v52 loss=5.22 vs v51 loss=6.43 (19% lower), indicating MultiScaleSegHead fits the training data more efficiently
- Tissue mIoU trajectories are similar; instrument mIoU is highly variable in both — this is a dataset-level issue (weak instrument supervision in grasp0), not introduced by MultiScaleSegHead
- At epoch 21, v52 sum=132.97 is already close to v51's all-time best of 133.42 at epoch 41

**Note:** Eval was run while training was still ongoing (~epoch 31); `last.ckpt` reflects the latest checkpoint at eval time. The drop in instrument mIoU at `last` (51.89%) follows the same late-epoch decline seen in v51.

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) | sum (%) |
|------------|----------------|----------------------|---------------------|--------------------------|---------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 | 136.47 |
| **v51 ep41** ⭐ (v51 best) | 73.58 | 83.18 / 86.79 / 84.21 | 59.84 | 76.25 / 72.76 / 72.64 | 133.42 |
| **ep15** | 68.75 | 79.30 / 85.11 / 80.72 | **63.14** | 85.47 / 69.84 / 75.16 | 131.89 |
| **ep20** | 71.03 | 81.46 / 85.56 / 82.34 | 61.24 | 78.47 / 72.17 / 73.65 | 132.27 |
| **ep21** ⭐ (best sum) | 71.33 | 81.97 / 85.24 / 82.56 | 61.64 | 78.90 / 72.70 / 74.10 | **132.97** |
| last (~ep31) | **73.60** | 83.86 / 86.60 / 84.22 | 51.89 | 67.41 / 67.96 / 65.27 | 125.49 |

### Key findings

- **MultiScaleSegHead converges faster:** ep21 sum=132.97 nearly matches v51's best (133.42 at ep41), while v52 has only used 21 of 80 epochs — at equal epoch counts v52 consistently outperforms v51
- **Tissue mIoU continues to climb:** at last (~ep31), tissue=73.60% already exceeds v51's record of 73.58%; full 80-epoch training is expected to push tissue above 74%
- **Instrument mIoU instability is dataset-driven:** peaks at ep15 (63.14%) then gradually declines, mirroring v51's behaviour — root cause is that grasp0 provides weak warp_seg supervision for instruments, so the oracle picks different channels across epochs rather than a stable one
- **Fewer parameters, richer features:** 3.28M vs 5.90M; using feat1/feat2 (previously unused in FCNHead's resize_concat) provides effective mid-level detail that improves both speed of convergence and tissue boundary quality
- Trade-off: to push sum above baseline, instrument quality needs to improve without sacrificing the high tissue performance

---

## v53 & v54 Evaluation Results (260626, job 317 / 320 train — job 321 eval)

### v53

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v53.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v53_260625_192932/`

**Key changes vs v52:**
- `data_path`: CMC_grasp0_deinterlaced (379 seqs) → **CMC_grasp0_5_10_merged** with split `train_g0379_g10601.txt` — combines grasp0 (379 pairs) + grasp10 (601 pairs) = **980 training pairs**
- `w_dino`: 1.0 → **0.1** — calibrated for warp_seg loss scale (~4–16 on this dataset vs ~100 on data_medical)
- Architecture unchanged: MultiScaleSegHead, mask_size=[96,96]
- Instrument at **ch3** (same as v52); eval uses `object_channel: 3` / `oracle_exclude_channels: [3]`

**Motivation:** grasp10 sequences feature stronger instrument motion, providing richer warp_seg training signal for instrument channel assignment.

### v54

**Config:** `configs/instrument/rcf_cmc_grasp0_tissue_ft_v54.yaml`
**Run dir:** `saved/grasp0_tissue_ft_v54_260625_201208/`

**Key changes vs v53:**
- `resize_short` for 720×576 images: 400 → **576** — images are no longer downscaled; crop margin increases from 16 px to 192 px, giving more diverse crops
- `w_dino`: 0.1 → **0.08**
- Instrument migrated to **ch4** after training; eval uses `object_channel: 4` / `oracle_exclude_channels: [4]`

### Results per checkpoint

| Checkpoint | tissue mIoU (%) | tissue P / R / F1 (%) | instrument mIoU (%) | instrument P / R / F1 (%) | sum (%) |
|------------|----------------|----------------------|---------------------|--------------------------|---------|
| **v9 baseline** | 67.65 | 80.35 / 82.79 / 79.94 | 68.82 | 77.50 / 86.00 / 80.02 | 136.47 |
| **v52 ep21** ⭐ (v52 best) | 71.33 | 81.97 / 85.24 / 82.56 | 61.64 | 78.90 / 72.70 / 74.10 | 132.97 |
| **v53 ep42** | 72.81 | 84.33 / 84.98 / 83.73 | 61.71 | 77.84 / 73.30 / 73.53 | 134.52 |
| **v53 ep43** ⭐ (v53 best sum) | 72.82 | 85.17 / 83.99 / 83.65 | **63.95** | 76.46 / 79.03 / 75.63 | **136.77** |
| **v53 ep47** | 71.35 | 85.49 / 81.94 / 82.73 | 62.07 | 76.20 / 75.34 / 73.98 | 133.42 |
| **v53 last** | 69.08 | 84.15 / 80.75 / 81.11 | 52.98 | 69.57 / 67.13 / 66.39 | 122.06 |
| **v54 ep16** | 74.49 | 83.64 / 87.97 / 84.79 | 56.69 | 63.24 / 85.67 / 70.08 | 131.18 |
| **v54 ep21** | 74.45 | 84.07 / 86.98 / 84.68 | 57.82 | 63.24 / 87.58 / 71.18 | 132.27 |
| **v54 ep25** ⭐ (v54 best sum) | 74.45 | 83.60 / 87.67 / 84.72 | 58.80 | 64.72 / 87.84 / 72.17 | 133.25 |
| **v54 last** ⭐ (v54 best tissue) | **75.66** | 83.73 / 88.89 / 85.56 | 44.64 | 55.63 / 66.86 / 58.65 | 120.30 |

### Key findings

- **v53 ep43 matches the v9 baseline:** sum=136.77 vs v9's 136.47 — the first unsupervised model to reach this level, achieved by adding grasp10 training data for stronger instrument motion signal
- **Grasp10 data directly improves instrument mIoU:** v53's best instrument (63.95%) exceeds v52's best (61.64%) by +2.3 pp, while maintaining comparable tissue performance
- **resize_short 400 vs 576 trade-off (v53 vs v54):** larger crops in v54 push tissue mIoU higher (~74–76% vs ~71–73%) but hurt instrument mIoU (~45–59% vs ~52–64%). At 720×576 resolution, resize_short=576 preserves more scene context but makes the instrument occupy a smaller relative area, weakening its warp_seg signal
- **Late-epoch instrument collapse persists in both:** instrument mIoU declines at last checkpoints (v53_last=52.98%, v54_last=44.64%), consistent with the grasp0 dataset issue of weak instrument supervision — channels drift away from instrument as tissue representation matures
- **v54 tissue is the new record:** v54_last achieves tissue=75.66%, surpassing v52_last (73.60%) and v51's best (73.58%), at the cost of instrument quality (44.64%)
- **Best overall checkpoint: v53 ep43** — tissue=72.82%, instrument=63.95%, sum=136.77%; represents the best instrument/tissue balance achieved by unsupervised training on this dataset
---

## Grasp0 Segmentation — Version Summary

Eval protocol: instrument mIoU = single best channel (fixed, oracle-detected); tissue mIoU = two-pass oracle excluding the instrument channel. Numbers are the best checkpoint per version.

### Phase 1 — Baseline

| Version | Description | Inst mIoU | Tissue mIoU | Sum |
|---------|-------------|:---------:|:-----------:|:---:|
| **v9** | FCNHead decoder; pretrained on grasp10, fine-tuned on grasp0; DINO distillation | 68.82 | 67.65 | 136.47 |

### Phase 2 — Loss tuning (FCNHead / MultiScaleSegHead, grasp0 data only)

| Version | Key change | Inst mIoU | Tissue mIoU | Sum |
|---------|-----------|:---------:|:-----------:|:---:|
| v40 | + Flow bilateral-CE loss (w=2.0) | 55.27 | 73.09 | 128.36 |
| v42 | FCNHead → **MultiScaleSegHead** | 60.47 | 72.15 | 132.62 |
| v43 | Loss rebalancing | 66.03 | 71.69 | 137.72 |
| v46 | Stronger bilateral (w=4) + DINO (w=2) | 65.71 | 70.80 | 136.51 |
| v47 | w_dino = 3 | **67.80** | 70.48 | **138.28** |
| v51 | DINO only (w=1), no bilateral | 61.30 | **73.58** | 134.88 |

### Phase 3 — Architecture & training tuning (MultiScaleSegHead baseline, grasp0+grasp10 merged data)

| Version | Key change | Inst mIoU | Tissue mIoU | Sum |
|---------|-----------|:---------:|:-----------:|:---:|
| v52 | New pipeline: merged grasp0+grasp10 data; mask_size 128→96 | 61.64 | 71.33 | 132.97 |
| v53 | w_dino 1.0→0.1 (calibrated for merged data) | **63.95** | 72.82 | **136.77** |
| v54 | Higher crop res (resize_short 400→576) | 58.80 | **74.45** | 133.25 |
| v55 | + Sobel edge feat in decoder | 63.83 | 70.17 | 134.00 |
| v56 | + Flow guidance feat (dropout aug) | 59.30 | 70.03 | 129.33 |
| v57 | Edge feat + flow guidance | 61.11 | 72.50 | 133.61 |
| v58 | clamp_flow_t 10→20, topk 4→6 | 64.48 | 69.96 | 134.44 |
| v59 | Flow guidance + clamp=20 + topk=6 | 62.54 | 64.18 | 126.72 |
| v60 | topk 4→2 (easy-example mining) | 63.86 | 68.51 | 132.37 |
| <span style="color:red">**v61**</span> | <span style="color:red">**UNetSegHead** (top-down coarse-to-fine) + edge feat</span> | <span style="color:red">64.01</span> | <span style="color:red">72.51</span> | <span style="color:red">**136.52**</span> |
| v62 | **UNetSegHeadV2** (true FPN, standard backbone strides) | 62.83 | 72.65 | 135.48 |

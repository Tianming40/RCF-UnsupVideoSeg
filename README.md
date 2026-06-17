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

## Loss schedule overview (v21–v22)

> **Key shift**: Replace `L_flow_tv` (gradient dead zone confirmed in v19/v20) with `L_flow_cluster_ce` (K-means color-block CE, external targets).
>
> **Core insight**: `L_flow_cosine` is self-referential — its CE target μ_c is derived from the current mask, so it reinforces the pretrained prior rather than breaking it. `L_flow_cluster_ce` uses external RAFT K-means labels (independent of current mask) and is therefore the true prior-breaking force for aligning non-instrument channels with RAFT flow color blocks.

| Ver | Ep | L_entropy | L_deform | L_distill | L_flow_cosine | fc: temp / diversity | L_flow_cluster_ce | fcc: temp / diversity / start_ep | Role balance |
|-----|-----|----------|---------|---------|-------------|-------------------|-----------------|--------------------------------|-------------|
| v21 | 80 | 0.05 | 0.5 | 1.0 relu | 0.5 | 0.5 / 0.5 | 0.5 | 0.3 / 0.5 / ep5 | equal weight; cluster_ce delayed to ep5 |
| v22 | 80 | 0.05 | 0.5 | 1.0 relu | **0.2** | 0.5 / **0.3** | **1.0** | 0.3 / 0.5 / **ep0** | cluster_ce primary (2×); cosine auxiliary (prevent collapse) |

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
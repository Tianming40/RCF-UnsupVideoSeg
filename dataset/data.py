import json
import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import random

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, training, frame_num=2, load_flow=False, load_pl=False, transform=None, subsample_frame_interval=None, flow_suffix="",flow_suffix2="",flow_suffix3="", zero_ann=False, pl_root=None, pl_root2=None, grasp_ann_dir=None,
                 gap_options=None, gap_probabilities=None, gap_flow_suffixes=None):
        super().__init__()

        # Random-gap sampling (see __getitem__): which frame-gap values a
        # training pair can be drawn at, with what probability, and which
        # Flows(+suffix)/BackwardFlows(+suffix) directory to read for each.
        # Default (None) reproduces the original hardcoded gap-1/2/3
        # behavior EXACTLY (same options/probabilities, same 3 suffix slots
        # sourced from flow_suffix/flow_suffix2/flow_suffix3) — every
        # existing caller (data_medical configs, CMC pair-dir configs) is
        # unaffected. Pass all three explicitly (matching lengths, probs
        # summing to 1) to use more/fewer/different gap values, e.g. the
        # CMC grasp0 multi-gap sequences which have real flow for gap 1..7.
        self.gap_options = gap_options if gap_options is not None else [1, 2, 3]
        self.gap_probabilities = gap_probabilities if gap_probabilities is not None else [0.7, 0.2, 0.1]
        self.gap_flow_suffixes = gap_flow_suffixes if gap_flow_suffixes is not None else [flow_suffix, flow_suffix2, flow_suffix3]
        assert len(self.gap_options) == len(self.gap_probabilities) == len(self.gap_flow_suffixes), \
            "gap_options / gap_probabilities / gap_flow_suffixes must have matching lengths"
        assert abs(sum(self.gap_probabilities) - 1.0) < 1e-6, "gap_probabilities must sum to 1"
        assert 1 in self.gap_options, "gap=1 must always be an option (used as the boundary-overflow fallback)"

        file_path = os.path.join(root, split)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines.sort()
        seq_lens = []
        seq_names = []
        seq_frames_path_all = []
        if subsample_frame_interval is not None:
            assert not training, "subsample_frame_interval is only for evaluation"
        for line in lines:
            line = line.split()
            seq_name = line[0]
            seq_frames = line[1:]
            if subsample_frame_interval == -1:
                # First frame for every sequence
                seq_frames = seq_frames[:1]
            elif subsample_frame_interval is not None:
                seq_frames = seq_frames[::subsample_frame_interval]
            seq_lens.append(len(seq_frames))
            seq_names.append(seq_name.rstrip("/").split("/")[-1])
            seq_frames_path_all.append(
                [os.path.join(root, seq_name, frame_filename) for frame_filename in seq_frames])

        self.seq_names = seq_names
        self.seq_frames_path_all = seq_frames_path_all

        self.seq_lens = seq_lens
        self.seq_freq = seq_lens / np.sum(seq_lens)
        self.seq_len_cumsum = np.insert(np.cumsum(seq_lens), 0, 0)
        # print(self.seq_len_cumsum)
        self.num_seq = len(seq_lens)

        self.transform = transform

        self.frame_num = frame_num
        self.training = training
        self.load_flow = load_flow
        self.load_pl = load_pl
        self.flow_suffix = flow_suffix
        self.flow_suffix2 = flow_suffix2
        self.flow_suffix3 = flow_suffix3
        self.pl_root = pl_root
        self.pl_root2 = pl_root2

        self.zero_ann = zero_ann

        # Grasping / dissection point annotations {seq_name: (dissect_xy, grasp_xy)}
        # Coordinates are normalised [0, 1].  (-1, -1) means no annotation.
        self.grasp_annotations = {}
        if grasp_ann_dir is not None:
            for fname in os.listdir(grasp_ann_dir):
                if fname.startswith('_') or not fname.endswith('.json'):
                    continue
                seq = fname[:-5]
                try:
                    d = json.load(open(os.path.join(grasp_ann_dir, fname)))
                    pts = list(d['annotations'][0].values())[0]  # first annotation
                    self.grasp_annotations[seq] = (pts[0], pts[1])  # (dissect, grasp)
                except Exception:
                    pass

        if self.load_pl:
            assert self.transform.has_pl, "load_pl needs to match with has_pl in transform"

        if not self.training:
            # frame_num==1 is the normal case (scored frame only). frame_num>1
            # is allowed for eval variants that need an auxiliary neighbour
            # frame available at inference time (e.g. RCFJointMaskSoftTissueModel,
            # models/rcf_joint_mask_model.py) -- only current_seq[frame_ind]
            # (the first frame) is ever scored/annotated; extra frames are
            # loaded but never receive their own annotation lookup (see the
            # `assert i == 0` site below, relaxed for the same reason).
            assert self.frame_num >= 1, f"frame_num must be >= 1, got {self.frame_num}"

    def load_image(self, path, convert_format="RGB"):
        with open(path, "rb") as f:
            try:
                img = Image.open(f)
            except Exception as e:
                print("Error in loading image: ", e)
                img = Image.open(f)
            return img.convert(convert_format)

    def __getitem__(self, index):
        # subset is taken in the split txt file
        seq_ind_within_subset = np.digitize(index, self.seq_len_cumsum) - 1

        frame_ind = index - self.seq_len_cumsum[seq_ind_within_subset]

        # We don't get the last `self.frame_num - 1` frame(s) since we need current and next frame
        if frame_ind >= self.seq_lens[seq_ind_within_subset] - (self.frame_num - 1):
            # This shift-back makes __len__ (== sum(seq_lens), not sum(seq_lens
            # - (frame_num-1))) slightly overcount valid starts for any line
            # with frame_num>1 -- harmless for training (just an extra
            # stochastic duplicate draw across epochs, always allowed here)
            # and, since dataset/data.py's eval mode was relaxed this session
            # to allow frame_num>1 (see the __init__ assert above,
            # RCFJointMaskSoftTissueModel's paired eval), also harmless for
            # eval: the duplicate index resolves to the SAME frame_ind as an
            # earlier index, so it just re-scores an already-scored frame
            # (wasted compute, not a correctness issue -- doesn't skew miou).
            frame_ind -= self.frame_num - 1

        current_seq = self.seq_frames_path_all[seq_ind_within_subset]

        # images = []
        # for i in range(self.frame_num):
        #     path = current_seq[frame_ind + i]
        #     image = self.load_image(path)
        #     images.append(image)
        
        # Randomly select whether to acquire the next frame or two frames apart
        frame_gap = np.random.choice(self.gap_options, p=self.gap_probabilities)  # 1 means next frame, 2 means two frames apart.

        images = []
        flag_gap = 0
        for i in range(self.frame_num):
            # Calculate the index of the frame to be acquired
            frame_to_get = frame_ind + i * frame_gap

            # Make sure the index is within legal limits
            # If the requirement of two frames apart cannot be met, adjacent frames are used
            if frame_to_get >= len(current_seq):
                frame_to_get = frame_ind + i  # Using adjacent frames
                flag_gap = 1
            else:
                frame_to_get = min(frame_to_get, len(current_seq) - 1)  # Using Interval Frames
                flag_gap = frame_gap

            path = current_seq[frame_to_get]
            image = self.load_image(path)
            images.append(image)

        seq_name = self.seq_names[seq_ind_within_subset]

        ann_entry = self.grasp_annotations.get(seq_name, None)
        if ann_entry is not None:
            dissect_xy = torch.tensor(ann_entry[0], dtype=torch.float32)  # [2]  (x,y) normalised
            grasp_xy   = torch.tensor(ann_entry[1], dtype=torch.float32)  # [2]
        else:
            dissect_xy = torch.tensor([-1.0, -1.0], dtype=torch.float32)
            grasp_xy   = torch.tensor([-1.0, -1.0], dtype=torch.float32)

        ret = {
            'imgs': images,
            'seq_ids': seq_ind_within_subset,
            'seq_names': seq_name,
            'paths': current_seq[frame_ind:frame_ind+self.frame_num],
            'frame_ind_start': frame_ind,
            'seg_fields': [],
            'grasp_xy':   grasp_xy,    # normalised (x,y), (-1,-1) if unavailable
            'dissect_xy': dissect_xy,  # normalised (x,y), (-1,-1) if unavailable
            'gap': flag_gap,  # actual frame gap used for this sample (post boundary-fallback) — always 1 for eval/frame_num=1
        }

        if not self.training:
            # Annotation always belongs to the FIRST frame (current_seq[frame_ind]),
            # regardless of frame_num -- extra frames (frame_num>1, see the
            # __init__ assert above) are auxiliary and never separately annotated.
            if not self.zero_ann:
                path = current_seq[frame_ind].replace(
                    "JPEGImages", "Annotations").replace(".png", ".jpg")    # rewrite by wpr
                ann = self.load_image(path)
            else:
                # Set ann to 1x1 zeros
                ann = Image.fromarray(np.array([[[0, 0, 0]]], dtype=np.uint8))
            # Do not resize annotations (not adding into seg_fields): mask will be resized to annotations
            # Support one annotation for now
            ret['ann'] = ann

        if self.load_flow:
            # flag_gap always lands on one of self.gap_options (fallback path
            # above forces it to 1, which is always index 0 by construction —
            # see the assert in __init__). Each gap value reads flow from its
            # own Flows(+suffix)/BackwardFlows(+suffix) directory, keyed by
            # the SECOND frame's path (matches data_medical's Flows_NewCT2/
            # 0002.npy convention: the flow file is named after its target
            # frame, which uniquely determines the source frame given a
            # fixed gap).
            flow_suffix_for_gap = self.gap_flow_suffixes[self.gap_options.index(flag_gap)]
            gt_fw_flows = []
            gt_bw_flows = []
            for i in range(1, self.frame_num): # 00001.jpg in Flow is the flow from 0 to 1
                fw_flow_path = current_seq[frame_ind + i * flag_gap].replace(
                    "JPEGImages", "Flows" + flow_suffix_for_gap)[:-4] + ".npy"
                bw_flow_path = current_seq[frame_ind + i * flag_gap].replace(
                    "JPEGImages", "BackwardFlows" + flow_suffix_for_gap)[:-4] + ".npy"
                gt_fw_flow = np.load(fw_flow_path)
                gt_bw_flow = np.load(bw_flow_path)

                ### Data format modification
                gt_fw_flow = gt_fw_flow.astype(np.float32)
                gt_bw_flow = gt_bw_flow.astype(np.float32)

                gt_fw_flows.append(gt_fw_flow)
                gt_bw_flows.append(gt_bw_flow)

            ret['gt_fw_flows'] = gt_fw_flows
            ret['gt_bw_flows'] = gt_bw_flows
            ret['seg_fields'].extend(['gt_fw_flows', 'gt_bw_flows'])
            

        if self.load_pl:
            # PL is different from annotaiton as it requires augmentation
            
            
            # pl_masks_1
            pl_masks_1 = []
            for i in range(self.frame_num):
                # frame_ind + i
                img_filename = current_seq[frame_ind + i].split('/')[-1][:-4]
                path = os.path.join(self.pl_root, f'pred_seg_{seq_name}_{img_filename}_0000000.png')
                pl_mask_1 = np.asarray(self.load_image(path, convert_format="L"))
                pl_masks_1.append(pl_mask_1)
            ret['pl_masks_1'] = pl_masks_1
            ret['seg_fields'].append('pl_masks_1')
            
            # pl_masks_2
            pl_masks_2 = []
            for j in range(self.frame_num):
                # frame_ind + j
                img_filename2 = current_seq[frame_ind + j].split('/')[-1][:-4]
                path2 = os.path.join(self.pl_root2, f'pred_seg_{seq_name}_{img_filename2}_0000000.png')
                pl_mask_2 = np.asarray(self.load_image(path2, convert_format="L"))
                pl_masks_2.append(pl_mask_2)
            ret['pl_masks_2'] = pl_masks_2
            ret['seg_fields'].append('pl_masks_2')
                                   

        if self.transform is not None:
            ret = self.transform(ret)

        # When collated, imgs will become `self.frame_num` arrays. Same for paths.
        return ret

    def __len__(self):
        # Debugging:
        # return 8
        return np.sum(self.seq_lens)


if __name__ == "__main__":
    np.random.seed(1)
    
    dataset = VideoDataset('../data/data_SegTrackv2_resized', training=True, load_flow=True, split='trainval.txt', flow_suffix="_NewCT")

    for item in dataset:
        continue

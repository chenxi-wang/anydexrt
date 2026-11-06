"""Retargeting training dataset in the robot hand coordinate system.
Author: Chenxi Wang, Ying Feng
"""
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from anydexrt.geometry.poses import manus2fingertip

class HandGlovePairDataset(Dataset):
    def __init__(self, root,hand, num_repeat=100,  unified_norm=False):
        super().__init__()
        self.root = root
        if num_repeat < 1:
            raise ValueError("num_repeat must be positive")
        self.num_repeat = num_repeat
        self.unified_norm = unified_norm
        self.hand = hand
        self.hand_finger_pose = []
        self.glove_finger_pose = []
        self.finger_idxs = []
        self.hand_min_value = None
        self.hand_max_value = None
        self.glove_min_value = None
        self.glove_max_value = None
        self.glove_scales = [1.0] * self.hand.finger_num
        self.hand_scales = [1.0] * self.hand.finger_num
        self.anchor_data = {
            "hand": {i: None for i in range( self.hand.finger_num)},
            "glove": {i: None for i in range( self.hand.finger_num)}
        }
        
        self.load_all_data()

    def __len__(self):
        return len(self.glove_finger_pose) * self.num_repeat

    def __getitem__(self, index):
        index %= len(self.glove_finger_pose)
        glove_finger_pose = self.glove_finger_pose[index].copy()
        finger_idx = self.finger_idxs[index]
        glove_data = self.random_finger_completion(glove_finger_pose, finger_idx)
        glove_data = self.aug_glove_data(glove_data)
        return glove_data.astype(np.float32)
    
    def aug_glove_data(self, glove_data, offset_range=0.002):
        glove_data = glove_data.copy()
        offsets = np.random.uniform(-offset_range, offset_range, size=glove_data.shape)
        glove_data_aug = glove_data + offsets
        glove_data_aug[..., 3:7] = glove_data[..., 3:7]
        return glove_data_aug

    def load_all_data(self):
        hand_dir = os.path.join(self.root, "generated_hand_pose")
        glove_dir = os.path.join(self.root, "glove_data")

        with open(os.path.join(hand_dir, "norm_stats.json"), "r") as f:
            meta = json.load(f)
            self.hand_min_value = np.array(meta["min_value"], dtype=np.float32)
            self.hand_max_value = np.array(meta["max_value"], dtype=np.float32)
        with open(os.path.join(glove_dir, "norm_stats.json"), "r") as f:
            meta = json.load(f)
            self.glove_min_value = np.array(meta["min_value"], dtype=np.float32)
            self.glove_max_value = np.array(meta["max_value"], dtype=np.float32)
        if self.unified_norm:
            self.hand_min_value[:] = self.hand_min_value.min(axis=0)
            self.hand_max_value[:] = self.hand_max_value.max(axis=0)
            self.glove_min_value[:] = self.glove_min_value.min(axis=0)
            self.glove_max_value[:] = self.glove_max_value.max(axis=0)

        self.per_finger_glove = []
        for i in tqdm(range( self.hand.finger_num), desc="Loading hand data ..."):
            hand_path = os.path.join(hand_dir, f"finger{i}.npz")
            glove_path = os.path.join(glove_dir, f"finger{i}.npz")
            hand_data = np.load(hand_path)["pose"].astype(np.float32)
            glove_data = np.load(glove_path)["finger_pose"].astype(np.float32)

            if len(hand_data) == 0 or len(glove_data) == 0:
                raise ValueError(f"Empty motion data for finger {i}")
            anchor_hand = []
            anchor_glove = []
            for pattern in ["bend", "lateral"]:
                hand_path = os.path.join(self.root, "anchor", "hand_fingertip", f"finger{i}_{pattern}.npy")
                glove_path = os.path.join(self.root, "anchor", "glove_keypoint", f"finger{i}_{pattern}.npy")
                if not os.path.exists(hand_path):
                    continue
                if not os.path.exists(glove_path):
                    continue
                anchor_hand.append(np.load(hand_path)[:, i][:,:3])
                anchor_glove_kp = np.load(glove_path)
                for kps in anchor_glove_kp:
                    anchor_glove.append(manus2fingertip(kps)[self.hand.hand2manus[i]]) 

            if not anchor_hand or len(anchor_glove) != sum(len(a) for a in anchor_hand):
                raise ValueError(f"Missing or unpaired anchors for finger {i}")
            anchor_hand = np.concatenate(anchor_hand, axis=0).astype(np.float32)
            anchor_glove = np.stack(anchor_glove, axis=0).astype(np.float32)

            self.anchor_data["hand"][i] = anchor_hand
            self.anchor_data["glove"][i] = anchor_glove

            hand_center_value = ((self.hand_max_value + self.hand_min_value) / 2)[i, :3] # [3]
            hand_grid_length = np.max((self.hand_max_value - self.hand_min_value)[i, :3]) # scalar
            if hand_grid_length <= 0:
                raise ValueError(f"Zero robot position range for finger {i}")
            hand_data[:, :3] = (hand_data[:, :3] - hand_center_value) / hand_grid_length * 2
            self.anchor_data["hand"][i][:, :3] = (self.anchor_data["hand"][i][:, :3] - hand_center_value) / hand_grid_length * 2
            self.hand_scales[i] = hand_grid_length

            glove_center_value = ((self.glove_max_value + self.glove_min_value) / 2)[i, :3] # [3]
            glove_grid_length = np.max((self.glove_max_value - self.glove_min_value)[i, :3]) # scalar
            if glove_grid_length <= 0:
                raise ValueError(f"Zero glove position range for finger {i}")
            glove_data[:, :3] = (glove_data[:, :3] - glove_center_value) / glove_grid_length * 2
            self.anchor_data["glove"][i][:, :3] = (self.anchor_data["glove"][i][:, :3] - glove_center_value) / glove_grid_length * 2
            self.glove_scales[i] = glove_grid_length

            num_sample = len(glove_data)
            self.hand_finger_pose.append(hand_data)
            self.glove_finger_pose.append(glove_data)
            self.per_finger_glove.append(glove_data)
            self.finger_idxs += [i] * num_sample

        self.glove_finger_pose = np.concatenate(self.glove_finger_pose)
        self.glove_scales = np.array(self.glove_scales, dtype=np.float32)
        self.hand_scales = np.array(self.hand_scales, dtype=np.float32)

    def random_finger_completion(self, finger_pose, finger_idx):
        glove_data = np.stack([
            poses[np.random.randint(len(poses))] for poses in self.per_finger_glove
        ])
        glove_data[finger_idx] = finger_pose
        return glove_data

    def sample_batch_anchor_data(self, batch_size):
        anchor_glove = []
        anchor_hand = []
        for i in range(self.hand.finger_num):
            idxs = np.random.choice(len(self.anchor_data["hand"][i]), batch_size, replace=True)
            anchor_hand.append(self.anchor_data["hand"][i][idxs])
            anchor_glove.append(self.anchor_data["glove"][i][idxs])
        anchor_hand = np.stack(anchor_hand, axis=1)
        anchor_glove = np.stack(anchor_glove, axis=1)
        return anchor_glove, anchor_hand

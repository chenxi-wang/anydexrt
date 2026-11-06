"""Data normalization.
Author: Chenxi Wang, Ying Feng
"""
import os
import json
import numpy as np

def normalize(data, min_value, max_value):
    """data: [N, finger_num, 3/7], min/max_value: [finger_num, 3/7]"""
    data = np.array(data, dtype=np.float32, copy=True)
    center_value = ((min_value + max_value) / 2)[:, :3] # [finger_num, 3]
    grid_length = np.max((max_value - min_value)[:, :3], axis=-1) # [finger_num]
    if np.any(grid_length <= 0):
        raise ValueError("Normalization requires nonzero position range for every finger")
    data[..., :3] = (data[..., :3] - center_value) / grid_length[..., None] * 2
    return data

def unnormalize(data, min_value, max_value):
    """data: [N, finger_num, 3/7], min/max_value: [finger_num, 3/7]"""
    data = np.array(data, dtype=np.float32, copy=True)
    center_value = ((min_value + max_value) / 2)[:, :3] # [finger_num, 3]
    grid_length = np.max((max_value - min_value)[:, :3], axis=-1) # [finger_num]
    if np.any(grid_length <= 0):
        raise ValueError("Normalization requires nonzero position range for every finger")
    data[..., :3] = data[..., :3] * grid_length[..., None] / 2.0 + center_value
    return data

def load_norm_stats(data_dir, device_type="glove"):
    assert device_type in ["glove", "hand"]
    split = "glove_data" if device_type == "glove" else "generated_hand_pose"
    norm_path = os.path.join(data_dir, split, "norm_stats.json")
    with open(norm_path, "r") as f:
        meta = json.load(f)
        norm_min = np.array(meta["min_value"])
        norm_max = np.array(meta["max_value"])
    return norm_min, norm_max

"""Generate robot workspace samples and collect glove motion/contact datasets.
Author: Chenxi Wang, Ying Feng
"""
import os
import json
import time
import numpy as np
from tqdm import tqdm
from anydexrt.visualization.glove import GloveVisualizer
from anydexrt.geometry.poses import manus2fingertip, manus2fingers

def generate_finger_pose_joint_pair(hand_vis, finger_id, save_dir,config):
    joint_data = []
    pose_data = []
    reset_joint= np.zeros(hand_vis.joint_shape)

    if config["joint"]["joint"][finger_id]==1:
        items = config["joint"]["joint_ids"][str(finger_id)]
        for item in items:
            cid = item["cid"]
            mid = item["mid"]
            weight = item["weight"]
            offset = item["offset"]
            reset_joint[:, cid] = weight * reset_joint[:, mid] + offset
    for i in tqdm(range(10000)):
        joint = np.random.uniform(
            low=np.array(config["joint"]["limit"]["lower"])[finger_id],
            high=np.array(config["joint"]["limit"]["upper"])[finger_id]
        )
        if config["joint"]["joint"][finger_id]==1:
            items = config["joint"]["joint_ids"][str(finger_id)]
            for item in items:
                cid = item["cid"]
                mid = item["mid"]
                weight = item["weight"]
                offset = item["offset"]
                reset_joint[:, cid] = weight * reset_joint[:, mid] + offset
                joint[cid] = weight * joint[mid] + offset
        reset_joint[finger_id] = joint
        hand_vis.update(reset_joint)
        finger_pose = hand_vis.get_finger_pose()[finger_id]
        joint = hand_vis.get_joints()[finger_id]
        joint_data.append(joint)
        pose_data.append(finger_pose)

    joint_data = np.stack(joint_data, axis=0)
    pose_data = np.stack(pose_data, axis=0)
    save_path = os.path.join(save_dir, f"finger{finger_id}.npz")
    os.makedirs(save_dir, exist_ok=True)
    np.savez(save_path, joint=joint_data, pose=pose_data)

def _record_glove_samples(glove, finger_ids, sample_count, title):
    """Preview continuously; return only a complete keyboard-triggered episode."""
    from threading import Event
    from pynput import keyboard

    start_recording, cancel_recording = Event(), Event()

    def on_press(key):
        char = getattr(key, 'char', None)
        if char and char.lower() == 'r':
            start_recording.set()
        elif char and char.lower() == 'q':
            cancel_recording.set()

    viewer = GloveVisualizer(title, finger_ids)
    listener = keyboard.Listener(on_press=on_press)
    try:
        print(f"{title}: press 'r' to start, or 'q' to cancel. No terminal input is needed.")
        listener.start()
        while not start_recording.is_set():
            if cancel_recording.is_set() or not viewer.update(glove.get_hand_poses(format='raw')):
                raise KeyboardInterrupt('Collection cancelled; no partial recording saved')
            time.sleep(0.01)

        frames = []
        for _ in tqdm(range(sample_count), desc=title):
            if cancel_recording.is_set():
                raise KeyboardInterrupt('Collection cancelled; no partial recording saved')
            raw = glove.get_hand_poses(format='raw')
            if not viewer.update(raw) or cancel_recording.is_set():
                raise KeyboardInterrupt('Collection cancelled; no partial recording saved')
            frames.append(raw.copy())
            viewer.add_sample(raw)
            time.sleep(0.01)
        if cancel_recording.is_set():
            raise KeyboardInterrupt('Collection cancelled; no partial recording saved')
        return np.stack(frames)
    finally:
        listener.stop()
        if listener.ident is not None:
            listener.join(timeout=1)
        viewer.close()


def record_finger_data(glove, finger_id, save_dir):
    name = ('thumb', 'index', 'middle', 'ring', 'pinky')[finger_id]
    frames = _record_glove_samples(glove, [finger_id], 6000, f'Motion: {name}')
    finger_poses = np.stack([manus2fingertip(frame)[finger_id] for frame in frames])
    keypoints = np.stack([manus2fingers(frame) for frame in frames])
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f'finger{finger_id}.npz'),
             finger_pose=finger_poses, keypoint=keypoints)


def record_glove_data(glove, finger_ids, save_dir):
    frames = _record_glove_samples(glove, finger_ids, 3000, f'Contact: fingers {finger_ids}')
    finger_poses = np.stack([manus2fingertip(frame) for frame in frames])
    filename = '_'.join(str(i) for i in finger_ids)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f'finger_{filename}.npz'),
             finger_pose=finger_poses, finger_ids=np.asarray(finger_ids))


def compute_norm_stats(data_dir, data_source="glove_data",finger_num=5):
    assert data_source in ["glove_data", "generated_hand_pose"]
    key = "finger_pose" if data_source == "glove_data" else "pose"
    min_value = []
    max_value = []
    split_dir = os.path.join(data_dir, data_source)
    for i in range(finger_num):
        data_path = os.path.join(split_dir, f"finger{i}.npz")
        data = np.load(data_path)[key]
        min_value.append(data.min(axis=0).tolist())
        max_value.append(data.max(axis=0).tolist())
    os.makedirs(split_dir, exist_ok=True)
    save_path = os.path.join(split_dir, "norm_stats.json")
    save_dict = {
        "min_value": min_value,
        "max_value": max_value,
    }
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4)

"""Human anchor recording and their robot reference trajectories.
Author: Chenxi Wang, Ying Feng
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AnchorTask:
    name: str
    fingers: tuple[int, ...]
    pattern: str
    frames: int
    instruction: str


def anchor_tasks(config):
    """Use MANUS identity to find the thumb, regardless of robot finger order."""
    thumb = int(config['joint']['manus2hand_mapping']['0'])
    others = tuple(i for i in range(config['finger_num']) if i != thumb)
    return (
        AnchorTask('fingers_bend', others, 'bend', 100,
                   'Repeatedly open your four non-thumb fingers and make a fist. '
                   'Keep your thumb still. Start with the fingers open.'),
        AnchorTask('thumb_bend', (thumb,), 'bend', 100,
                   'Repeatedly bend and extend your thumb. Keep the other fingers still. '
                   'Start with the thumb extended.'),
        AnchorTask('thumb_lateral', (thumb,), 'lateral', 50,
                   'Repeatedly rotate your thumb laterally through its comfortable range. '
                   'Keep the other fingers still. Start at the lower end of the lateral range defined by your robot anchor configuration.'),
    )


def _apply_coupling(joints, config):
    """Apply each finger's coupling only to that finger's joint row."""
    for finger, coupled in enumerate(config['joint']['joint']):
        if coupled:
            for rule in config['joint']['joint_ids'][str(finger)]:
                joints[..., finger, rule['cid']] = (
                    rule['weight'] * joints[..., finger, rule['mid']] + rule['offset']
                )


def generate_anchor_trajectory(config, task, save_path):
    """Animate selected fingers and hold every other finger at its resting pose.

    Preserve the existing sweep sample counts and endpoint convention (i / N).
    """
    limits = config['joint'][f'anchor_{task.pattern}_limit']
    lower, upper = np.asarray(limits['lower']), np.asarray(limits['upper'])
    reset = np.array(config['joint']['anchor_bend_limit']['lower'], dtype=float)
    active = list(task.fingers)
    reset[active] = lower[active]
    _apply_coupling(reset, config)
    mask = np.zeros(config['joint']['shape'], dtype=bool)
    mask[active] = True
    replay = np.repeat(reset[None], task.frames, axis=0)
    ratio = np.arange(task.frames, dtype=float)[:, None, None] / task.frames
    replay[:, active] = (1 - ratio) * lower[active] + ratio * upper[active]
    _apply_coupling(replay, config)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, reset_joint=reset, replay_data=replay, joint_mask=mask)


def prepare_robot_anchors(config, local, hand):
    """Regenerate references and export the per-finger layout used by training.

    Named task references prevent reuse of old whole-hand thumb animations.
    """
    local = Path(local)
    tasks = anchor_tasks(config)
    for task in tasks:
        reference = local / 'anchor/robot_trajectories' / f'{task.name}.npz'
        generate_anchor_trajectory(config, task, reference)
        pose_path = local / 'anchor/hand_fingertip' / f'finger{task.fingers[0]}_{task.pattern}.npy'
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        transform_hand_joint_to_pose(hand, reference, pose_path)
        poses = np.load(pose_path)
        with np.load(reference) as data:
            reference_data = {key: data[key] for key in data.files}
        for finger in task.fingers:
            stem = f'finger{finger}_{task.pattern}'
            joint_path = local / 'anchor/hand_joint' / f'{stem}.npz'
            joint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(joint_path, **reference_data)
            np.save(local / 'anchor/hand_fingertip' / f'{stem}.npy', poses)
    return tasks


def transform_hand_joint_to_pose(hand_vis, in_path, out_path):
    data = np.load(in_path)
    joints = data["replay_data"] # Shape: (N, 5, 4)
    reset_joint = data["reset_joint"]  # Shape: (5, 4)
    joint_mask = data["joint_mask"]  # Shape: (5, 4)
    poses = []
    for joint in joints:
        joint[~joint_mask] = reset_joint[~joint_mask]
        hand_vis.update(joint)
        pose = hand_vis.get_finger_pose()  # Shape: (5, 7)
        poses.append(pose)
    poses = np.stack(poses, axis=0)  # Shape: (N, 5, 7)
    np.save(out_path, poses)


def record_glove_anchor(glove, replay_path, save_path, instruction=None, finger_ids=()):
    """Record live glove motion, then resample to the robot reference length."""
    import time
    from threading import Event
    from pynput import keyboard
    from anydexrt.visualization.glove import GloveVisualizer
    from anydexrt.geometry.trajectories import extract_trajectory

    with np.load(replay_path) as reference:
        output_length = len(reference['replay_data'])
    recording, finished = Event(), Event()

    def on_press(key):
        char = getattr(key, 'char', None)
        if char and char.lower() == 'r':
            recording.set()
        elif char and char.lower() == 'q':
            finished.set()

    viewer = GloveVisualizer(instruction or 'Glove anchors', finger_ids)
    listener = keyboard.Listener(on_press=on_press)
    frames = []
    try:
        if instruction:
            print(instruction)
        print("Press 'r' at the starting pose; repeat the motion, then 'q' to finish. "
              "Closing the window cancels without saving.")
        listener.start()
        while not finished.is_set():
            raw = glove.get_hand_poses(format='raw')
            if not viewer.update(raw):
                raise KeyboardInterrupt('Anchor collection cancelled; no recording saved')
            if recording.is_set() and not finished.is_set():
                frames.append(raw.copy())
                viewer.add_sample(raw)
            time.sleep(0.1)
        if len(frames) < 2:
            raise KeyboardInterrupt('Anchor collection cancelled; not enough samples to save')
        frames = np.asarray(frames)
        anchors = np.stack([
            extract_trajectory(frames[:, joint], output_length=output_length)
            for joint in range(25)
        ], axis=1)
        np.save(save_path, anchors)
    finally:
        listener.stop()
        if listener.ident is not None:
            listener.join(timeout=1)
        viewer.close()

"""Explicit collection stages.
Author: Chenxi Wang, Ying Feng
"""
from pathlib import Path
import numpy as np
from anydexrt.geometry.poses import pose_7d_to_4x4, pose_4x4_to_7d


def transform_poses(poses, hand, config):
    """Use the same base-frame convention for motion, anchors and inference."""
    base = np.asarray(config.get('base_transform', np.eye(4)))
    inv = np.linalg.inv(base)
    result = np.empty_like(poses)
    for idx in np.ndindex(poses.shape[:-1]):
        pose = base @ pose_7d_to_4x4(poses[idx]) @ inv
        result[idx] = hand.transform_glove_to_hand(pose_4x4_to_7d(pose))
    return result


def collect(config, data_root, stage, serial=None, address='tcp://127.0.0.1:8000', hand_type='right'):
    from .sampling import generate_finger_pose_joint_pair, compute_norm_stats, record_finger_data, record_glove_data
    from .anchors import prepare_robot_anchors, record_glove_anchor
    from anydexrt.visualization.hand import HandVisualizer
    from anydexrt.hardware.manus import MANUSGlove

    root = Path(data_root)
    local = root / config['hand_name']
    shared = root / 'global_data'
    hand = HandVisualizer(config)
    if stage == 'robot':
        generated = local / 'generated_hand_pose'
        generated.mkdir(parents=True, exist_ok=True)
        for i in range(config['finger_num']):
            if not (generated / f'finger{i}.npz').exists():
                generate_finger_pose_joint_pair(hand, i, generated, config)
        compute_norm_stats(local, 'generated_hand_pose', config['finger_num'])
        prepare_robot_anchors(config, local, hand)
        return
    if stage == 'anchors':
        glove = None

        def record_missing(_glove, *args, **kwargs):
            nonlocal glove
            if glove is None:
                if not serial:
                    raise ValueError('A glove serial is required to record missing human anchors')
                glove = MANUSGlove(serial, addr=address, hand_type=hand_type)
            return record_glove_anchor(glove, *args, **kwargs)

        try:
            collect_anchor_recordings(config, local, hand, None, record_missing, shared)
        finally:
            if glove is not None:
                glove.stop()
        return
    glove = None

    def get_glove(recording):
        nonlocal glove
        if glove is None:
            if not serial:
                raise ValueError(
                    f'--glove-serial is required to record missing data: {recording}'
                )
            glove = MANUSGlove(serial, addr=address, hand_type=hand_type)
        return glove

    try:
        if stage == 'motion':
            # All five MANUS fingers are needed by the contact model, even for a four-finger robot.
            for i in range(5):
                recording = shared / 'glove_data' / f'finger{i}.npz'
                if not recording.exists():
                    record_finger_data(get_glove(recording), i, shared / 'glove_data')
            for i in range(config['finger_num']):
                glove_id = hand.hand2manus[i]
                with np.load(shared / 'glove_data' / f'finger{glove_id}.npz') as f:
                    pose = transform_poses(f['finger_pose'], hand, config)
                    keypoint = transform_poses(f['keypoint'], hand, config)
                dest = local / 'glove_data' / f'finger{i}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez(dest, finger_pose=pose, keypoint=keypoint)
            compute_norm_stats(shared, 'glove_data', 5)
            compute_norm_stats(local, 'glove_data', config['finger_num'])
        elif stage == 'contact':
            dest = shared / 'contact_data'
            for i in range(1, 5):
                recording = dest / f'finger_0_{i}.npz'
                if not recording.exists():
                    record_glove_data(get_glove(recording), [0, i], dest)
        else:
            raise ValueError(f'Unknown collection stage: {stage}')
    finally:
        if glove is not None:
            glove.stop()


def _load_anchor_recording(path, frames):
    poses = np.load(path)
    if poses.shape != (frames, 25, 7) or not np.isfinite(poses).all():
        raise ValueError(f'{path} must contain {frames} finite 25-joint poses')
    return poses


def collect_anchor_recordings(config, local, hand, glove, recorder, shared):
    """Reuse shared human motions, then export anchors in the selected robot frame."""
    from .anchors import prepare_robot_anchors

    local = Path(local)
    tasks = prepare_robot_anchors(config, local, hand)
    for index, task in enumerate(tasks, start=1):
        raw = Path(shared) / 'anchor/raw_glove' / f'{task.name}.npy'
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            print(f'Anchor recording {index}/3: {task.instruction}')
            recorder(
                glove,
                local / 'anchor/robot_trajectories' / f'{task.name}.npz',
                raw, instruction=task.instruction,
                finger_ids=(1, 2, 3, 4) if task.name == 'fingers_bend' else (0,),
            )
        poses = _load_anchor_recording(raw, task.frames)
        transformed = transform_poses(poses, hand, config)
        target = local / 'anchor/glove_keypoint'
        target.mkdir(parents=True, exist_ok=True)
        # The same full-skeleton fist recording supplies all non-thumb fingers.
        # Training selects each finger using the robot-to-MANUS mapping.
        for finger in task.fingers:
            np.save(target / f'finger{finger}_{task.pattern}.npy', transformed)

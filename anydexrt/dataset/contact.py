"""Contact training dataset in the five-finger MANUS coordinate system.
Author: Chenxi Wang, Ying Feng
"""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .normalization import load_norm_stats, normalize


class ContactDataset(Dataset):
    """Recorded contacts and synthetic negatives from independent motion samples.

    Reject pairs whose thumb and target are both near known contact poses.
    Sampling raises a clear error if the motion data cannot supply a negative
    within the retry budget; no recorded-negative dataset is used.
    """
    def __init__(self, root, max_capacity=1000000, positive_ratio=0.6,
                 dist_thresh=0.002):
        if max_capacity < 1 or not 0 <= positive_ratio <= 1 or dist_thresh <= 0:
            raise ValueError('Invalid capacity, positive ratio, or distance threshold')
        self.max_capacity = max_capacity
        self.positive_ratio = positive_ratio
        self.dist_thresh = dist_thresh
        lo, hi = load_norm_stats(root)
        self.scales = np.max((hi - lo)[:, :3], axis=-1)
        self.motion = []
        self.positive = {}
        for i in range(5):
            with np.load(Path(root) / 'glove_data' / f'finger{i}.npz') as f:
                pose = f['finger_pose'].astype(np.float32)
            if len(pose) == 0:
                raise ValueError(f'Empty motion data for finger {i}')
            self.motion.append(normalize(pose[:, None], lo[i:i+1], hi[i:i+1])[:, 0, :3])
            if i:
                with np.load(Path(root) / 'contact_data' / f'finger_0_{i}.npz') as f:
                    poses = f['finger_pose']
                if poses.ndim != 3 or poses.shape[1:] != (5, 7) or len(poses) == 0:
                    raise ValueError('Contact recordings must have shape [N, 5, 7] with N > 0')
                poses = normalize(poses, lo, hi)
                self.positive[i] = np.concatenate([poses[:, 0, :3], poses[:, i, :3]], axis=-1)

    def __len__(self):
        return self.max_capacity

    def __getitem__(self, index):
        positive = index < self.max_capacity * self.positive_ratio
        pairs = []
        for finger in range(1, 5):
            if positive:
                pair = self.positive[finger][index % len(self.positive[finger])]
            else:
                pair = self._negative(finger)
            pairs.append(pair)
        return np.asarray(pairs, dtype=np.float32), np.full(4, positive, dtype=np.float32)

    def _negative(self, finger):
        known = self.positive[finger]
        for _ in range(100):
            thumb = self.motion[0][np.random.randint(len(self.motion[0]))]
            target = self.motion[finger][np.random.randint(len(self.motion[finger]))]
            td = np.linalg.norm(known[:, :3] - thumb, axis=-1) * self.scales[0] / 2
            fd = np.linalg.norm(known[:, 3:] - target, axis=-1) * self.scales[finger] / 2
            if td.min() > self.dist_thresh or fd.min() > self.dist_thresh:
                return np.concatenate([thumb, target])
        raise RuntimeError(
            f'Could not generate a negative for finger {finger} after 100 attempts; '
            'check motion coverage and the contact distance threshold.'
        )

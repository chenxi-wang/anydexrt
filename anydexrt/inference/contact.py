"""Predict contact probabilities and fuse robot fingertip poses during inference.
Author: Chenxi Wang, Ying Feng
"""
import numpy as np
import torch
from anydexrt.dataset.normalization import normalize


@torch.no_grad()
def predict_contact(model, poses, norm_min, norm_max):
    poses = np.asarray(poses)
    if poses.ndim == 2:
        poses = poses[None]
    poses = normalize(poses, norm_min, norm_max)[..., :3]
    pairs = np.stack([np.concatenate([poses[:, 0], poses[:, i]], axis=-1) for i in range(1, 5)], axis=1)
    model.eval()
    return model(torch.as_tensor(pairs, dtype=torch.float32, device=next(model.parameters()).device))


def fuse_contacts_confidence(output, scores, templates, manus2hand, threshold=0.5):
    """Interpolate each contact candidate, then confidence-average full poses.

    Args:
        output: (robot fingers, 3) model-predicted fingertip XYZ positions.
        scores: Four probabilities in MANUS index/middle/ring/pinky order.
        templates: Per-robot-finger lookup arrays of fingertip XYZ positions.
        manus2hand: Mapping from MANUS finger IDs (thumb=0) to robot IDs.
        threshold: Strict activation cutoff, in [0, 1).

    For each supported pair with p > threshold, a = (p-threshold)/(1-threshold).
    Find the thumb-table point nearest the ORIGINAL predicted pair midpoint.
    The pair's full-pose candidate retains all other fingers, and interpolates
    both pair positions as (1-a)*prediction + a*lookup_point. Return the average
    of those candidates weighted by a. Thus shared thumb corrections combine
    without depending on contact iteration order. Other fingers' corrections
    are also diluted by competing candidates, intentionally averaging full poses.
    No active pairs (or no robot thumb) returns the original prediction.

    This softens contact activation; discrete lookup selection can still jump.
    Inputs are never modified.
    """
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('contact threshold must be finite and in [0, 1)')
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (4,) or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError('contact scores must be four finite probabilities in [0, 1]')
    original = np.array(output, dtype=float, copy=True)
    if 0 not in manus2hand:
        return original
    thumb = manus2hand[0]
    reference = np.asarray(templates[thumb])
    candidates, weights = [], []
    for index in np.flatnonzero(scores > threshold):
        if index + 1 not in manus2hand:
            continue
        finger = manus2hand[index + 1]
        alpha = (scores[index] - threshold) / (1 - threshold)
        ids = [thumb, finger]
        midpoint = original[ids].mean(axis=0)
        target = reference[np.argmin(np.linalg.norm(reference - midpoint, axis=-1))]
        candidate = original.copy()
        candidate[ids] = (1 - alpha) * original[ids] + alpha * target
        candidates.append(candidate)
        weights.append(alpha)
    return np.average(candidates, axis=0, weights=weights) if candidates else original

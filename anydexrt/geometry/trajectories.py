"""Extract smooth, uniformly spaced anchor curves from recorded glove motions.
Author: Chenxi Wang, Ying Feng

Projects positions onto a best-fit plane, selects and smooths an outer convex-hull
path, then resamples it by arc length and maps it back to 3D. The initial recorded
position determines path direction. For poses with wxyz quaternions, estimates
orientations from nearby recorded samples and smooths them with consistent signs.
"""
import numpy as np

def extract_trajectory(raw_trajectory, output_length=100):
    """Extract a smooth, uniformly-sampled outer-edge curve.

    raw_trajectory: np.ndarray, [N, 3] or [N, 7] with quaternion in wxyz
    output_length: number of points in the returned curve
    """
    raw_trajectory = np.asarray(raw_trajectory, dtype=float)
    output_length = int(output_length)
    if raw_trajectory.ndim != 2 or raw_trajectory.shape[1] not in (3, 7):
        raise ValueError("raw_trajectory must have shape [N, 3] or [N, 7]")
    if output_length < 2:
        raise ValueError("output_length must be at least 2")

    finite_mask = np.isfinite(raw_trajectory).all(axis=1)
    valid_trajectory = raw_trajectory[finite_mask]
    points = valid_trajectory[:, :3]
    if len(points) < 2:
        raise ValueError("raw_trajectory must contain at least two finite points")

    origin, plane_axes, points_2d = _project_to_plane(points)
    start_2d, end_2d = _compute_ordered_endpoints(points_2d, points_2d[0])

    hull_indices = _convex_hull_indices(points_2d)
    if len(hull_indices) < 2:
        curve_2d = np.repeat(start_2d[None, :], output_length, axis=0)
    else:
        hull = points_2d[hull_indices]
        start_idx = int(np.argmin(np.linalg.norm(hull - start_2d, axis=1)))
        end_idx = int(np.argmin(np.linalg.norm(hull - end_2d, axis=1)))
        outer_path = _longer_hull_path(hull, start_idx, end_idx)
        outer_path[0] = start_2d
        outer_path[-1] = end_2d
        outer_path = _chaikin_smooth(outer_path, iterations=4)
        outer_path[0] = start_2d
        outer_path[-1] = end_2d
        curve_2d = _resample_2d_by_arclength(outer_path, output_length)

    curve_3d = origin + curve_2d @ plane_axes
    if raw_trajectory.shape[1] == 3:
        return curve_3d

    rotations = _extract_curve_rotations(points, valid_trajectory[:, 3:7], curve_3d)
    return np.column_stack((curve_3d, rotations))

def _project_to_plane(points):
    origin = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    plane_axes = vh[:2]
    points_2d = (points - origin) @ plane_axes.T
    return origin, plane_axes, points_2d

def _compute_ordered_endpoints(points_2d, raw_start_2d):
    center = np.mean(points_2d, axis=0)
    centered = points_2d - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    progress = points_2d @ axis

    low_mask = progress <= np.quantile(progress, 0.08)
    high_mask = progress >= np.quantile(progress, 0.92)
    endpoint_a = _farthest_from_center(points_2d, low_mask, center, fallback_idx=np.argmin(progress))
    endpoint_b = _farthest_from_center(points_2d, high_mask, center, fallback_idx=np.argmax(progress))

    if np.linalg.norm(endpoint_b - raw_start_2d) < np.linalg.norm(endpoint_a - raw_start_2d):
        endpoint_a, endpoint_b = endpoint_b, endpoint_a
    return endpoint_a, endpoint_b

def _farthest_from_center(points_2d, mask, center, fallback_idx):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return points_2d[int(fallback_idx)]
    distances = np.linalg.norm(points_2d[indices] - center, axis=1)
    return points_2d[indices[int(np.argmax(distances))]]

def _convex_hull_indices(points_2d):
    if len(points_2d) <= 1:
        return np.arange(len(points_2d))

    order = sorted(range(len(points_2d)), key=lambda i: (points_2d[i, 0], points_2d[i, 1], i))

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for idx in order:
        while len(lower) >= 2 and cross(points_2d[lower[-2]], points_2d[lower[-1]], points_2d[idx]) <= 0:
            lower.pop()
        lower.append(idx)

    upper = []
    for idx in reversed(order):
        while len(upper) >= 2 and cross(points_2d[upper[-2]], points_2d[upper[-1]], points_2d[idx]) <= 0:
            upper.pop()
        upper.append(idx)

    return np.array(lower[:-1] + upper[:-1], dtype=int)

def _longer_hull_path(hull, start_idx, end_idx):
    if start_idx == end_idx:
        return hull[[start_idx]].copy()

    n = len(hull)
    if start_idx < end_idx:
        path_a = hull[start_idx:end_idx + 1]
        path_b = np.vstack((hull[end_idx:], hull[:start_idx + 1]))[::-1]
    else:
        path_a = np.vstack((hull[start_idx:], hull[:end_idx + 1]))
        path_b = hull[end_idx:start_idx + 1][::-1]

    length_a = _curve_length(path_a)
    length_b = _curve_length(path_b)
    return path_a.copy() if length_a >= length_b else path_b.copy()

def _chaikin_smooth(points, iterations=3):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points.copy()

    smoothed = points.copy()
    for _ in range(iterations):
        new_points = [smoothed[0]]
        for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
            new_points.append(0.75 * p0 + 0.25 * p1)
            new_points.append(0.25 * p0 + 0.75 * p1)
        new_points.append(smoothed[-1])
        smoothed = np.asarray(new_points)
    return smoothed

def _curve_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

def _resample_2d_by_arclength(points, output_length):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 1e-12]
    points = points[keep]
    if len(points) == 1:
        return np.repeat(points, output_length, axis=0)

    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    sample_lengths = np.linspace(0.0, arclength[-1], output_length)
    resampled = np.empty((output_length, 2), dtype=float)
    for dim in range(2):
        resampled[:, dim] = np.interp(sample_lengths, arclength, points[:, dim])
    return resampled

def _extract_curve_rotations(raw_points, raw_quaternions, curve_points):
    raw_quaternions = _normalize_quaternions(raw_quaternions)
    rotations = np.empty((len(curve_points), 4), dtype=float)

    k = min(8, len(raw_points))
    for i, point in enumerate(curve_points):
        distances = np.linalg.norm(raw_points - point, axis=1)
        neighbor_indices = np.argpartition(distances, k - 1)[:k]
        neighbor_distances = distances[neighbor_indices]
        scale = np.median(neighbor_distances) + 1e-12
        weights = np.exp(-0.5 * (neighbor_distances / scale) ** 2)
        rotations[i] = _weighted_quaternion_average(raw_quaternions[neighbor_indices], weights)

    rotations = _make_quaternion_sequence_continuous(rotations)
    rotations = _smooth_quaternions(rotations, window_size=max(5, len(rotations) // 20))
    return rotations

def _normalize_quaternions(quaternions):
    quaternions = np.asarray(quaternions, dtype=float).copy()
    norms = np.linalg.norm(quaternions, axis=1)
    valid = norms > 1e-12
    if not np.all(valid):
        replacement = np.array([1.0, 0.0, 0.0, 0.0])
        quaternions[~valid] = replacement
        norms[~valid] = 1.0
    return quaternions / norms[:, None]

def _weighted_quaternion_average(quaternions, weights):
    weights = np.asarray(weights, dtype=float)
    if np.sum(weights) <= 0:
        weights = np.ones(len(quaternions), dtype=float)
    weights = weights / np.sum(weights)

    reference = quaternions[np.argmax(weights)]
    aligned = quaternions.copy()
    flip_mask = np.dot(aligned, reference) < 0
    aligned[flip_mask] *= -1.0

    matrix = np.zeros((4, 4), dtype=float)
    for quaternion, weight in zip(aligned, weights):
        matrix += weight * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(matrix)
    averaged = vectors[:, -1]
    if np.dot(averaged, reference) < 0:
        averaged *= -1.0
    return averaged / np.linalg.norm(averaged)

def _make_quaternion_sequence_continuous(quaternions):
    quaternions = _normalize_quaternions(quaternions)
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] *= -1.0
    return quaternions

def _smooth_quaternions(quaternions, window_size):
    quaternions = _make_quaternion_sequence_continuous(quaternions)
    window_size = int(window_size)
    if window_size < 3:
        return quaternions
    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    smoothed = np.empty_like(quaternions)
    for i in range(len(quaternions)):
        left = max(0, i - half)
        right = min(len(quaternions), i + half + 1)
        local = quaternions[left:right].copy()
        reference = quaternions[i]
        local[np.dot(local, reference) < 0] *= -1.0
        weights = np.hanning(len(local) + 2)[1:-1]
        smoothed[i] = _weighted_quaternion_average(local, weights)
    return _make_quaternion_sequence_continuous(smoothed)

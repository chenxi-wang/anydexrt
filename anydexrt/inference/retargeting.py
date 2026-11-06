"""Retargeting inference, point-cloud results and live visualization.
Author: Chenxi Wang, Ying Feng
"""
from pathlib import Path
import numpy as np
import torch
from anydexrt.dataset.normalization import load_norm_stats, normalize, unnormalize
from anydexrt.models.retargeting import MappingModel
from .contact import fuse_contacts_confidence


def load_model(config, checkpoint, device='cpu'):
    model = MappingModel(config['finger_num']).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


@torch.no_grad()
def predict(model, positions, glove_stats, hand_stats):
    positions = normalize(positions, *glove_stats)
    tensor = torch.as_tensor(positions, dtype=torch.float32, device=next(model.parameters()).device)
    output, _ = model(tensor)
    return unnormalize(output.cpu().numpy(), *hand_stats)


def visualize_predictions(model, data_dir, output_dir=None, batch_size=1024, visualize=True):
    """Display glove/robot/predicted point clouds and optionally export XYZ arrays."""
    model.eval()
    data_dir = Path(data_dir)
    gs, hs = load_norm_stats(data_dir), load_norm_stats(data_dir, 'hand')
    results = {}
    for finger in range(model.finger_num):
        with np.load(data_dir / 'glove_data' / f'finger{finger}.npz') as f:
            glove = f['finger_pose']
        with np.load(data_dir / 'generated_hand_pose' / f'finger{finger}.npz') as f:
            hand = f['pose']
        predictions = []
        for j in range(0, len(glove), batch_size):
            inputs = np.repeat(glove[j:j+batch_size, None, :3], model.finger_num, axis=1)
            predictions.append(predict(model, inputs, gs, hs)[:, finger])
        pred = np.concatenate(predictions)
        results[f'finger{finger}'] = pred
        if output_dir:
            dest = Path(output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / f'finger{finger}.npy', pred)
        if visualize:
            import open3d as o3d
            clouds = []
            for points, color in [(glove[:, :3], [1,0,0]), (hand[:, :3], [0,1,0]), (pred, [0,0,1])]:
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                cloud.paint_uniform_color(color)
                clouds.append(cloud)
            o3d.visualization.draw_geometries(clouds)
    return results


def live(config, model, data_dir, serial, address, hand_type='right', contact_model=None, contact_data_dir=None,
         contact_threshold=0.5):
    if not np.isfinite(contact_threshold) or not 0 <= contact_threshold < 1:
        raise ValueError('contact threshold must be finite and in [0, 1)')
    import time
    import open3d as o3d
    from anydexrt.hardware.manus import MANUSGlove
    from anydexrt.visualization.hand import HandVisualizer
    from anydexrt.geometry.filtering import BatchOneEuroFilter
    from anydexrt.collection.pipeline import transform_poses
    from .contact import predict_contact

    hand = HandVisualizer(config)
    data_dir = Path(data_dir)
    gs, hs = load_norm_stats(data_dir), load_norm_stats(data_dir, 'hand')
    templates, joints, motion = [], [], []
    for finger in range(model.finger_num):
        with np.load(data_dir / 'generated_hand_pose' / f'finger{finger}.npz') as f:
            templates.append(f['pose'][:, :3])
            joints.append(f['joint'])
        with np.load(data_dir / 'glove_data' / f'finger{finger}.npz') as f:
            motion.append(f['finger_pose'][:, :3])
    contact_stats = load_norm_stats(contact_data_dir) if contact_model is not None else None
    smoother = BatchOneEuroFilter(config['joint']['num'], mincutoff=0.0, beta=0.5)
    glove = MANUSGlove(serial, addr=address, hand_type=hand_type)
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window()
        geoms = hand.render_hand()
        for geom in geoms:
            vis.add_geometry(geom)
        while vis.poll_events():
            raw = glove.get_hand_poses(format='fingertip')
            poses = raw[[hand.hand2manus[i] for i in range(model.finger_num)]]
            positions = transform_poses(poses, hand, config)[:, :3]
            for i, reference in enumerate(motion):
                if np.any(positions[i] < reference.min(axis=0)) or np.any(positions[i] > reference.max(axis=0)):
                    positions[i] = reference[np.argmin(np.linalg.norm(reference - positions[i], axis=-1))]
            output = predict(model, positions, gs, hs)[0]
            if contact_model is not None:
                scores = predict_contact(contact_model, raw, *contact_stats)[0].cpu().numpy()
                output = fuse_contacts_confidence(
                    output, scores, templates, hand.manus2hand, contact_threshold)
            selected = []
            for position, reference, joint in zip(output, templates, joints):
                distances = np.linalg.norm(reference - position, axis=-1)
                near = np.flatnonzero(distances <= 0.005)
                index = near[np.argmin(np.linalg.norm(joint[near], axis=-1))] if len(near) else distances.argmin()
                selected.append(joint[index])
            hand.update(smoother(np.asarray(selected).ravel()))
            for geom in geoms:
                vis.update_geometry(geom)
            vis.update_renderer()
            time.sleep(0.01)
    finally:
        glove.stop()
        vis.destroy_window()

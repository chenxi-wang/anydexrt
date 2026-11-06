"""Live MANUS skeleton rendering in the glove coordinate frame.
Author: Chenxi Wang, Ying Feng
"""
import numpy as np
import open3d as o3d


class GloveVisualizer:
    def __init__(self, title='MANUS glove', finger_ids=()):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=1280, height=720)
        self.finger_ids = tuple(finger_ids)
        self.trajectory = o3d.geometry.PointCloud()
        self.trajectory_added = False
        self.joints = o3d.geometry.PointCloud()
        self.bones = o3d.geometry.LineSet()
        edges = [(i, i + 1) for i in range(4)]
        for start in (5, 10, 15, 20):
            edges.append((0, start))
            edges.extend((i, i + 1) for i in range(start, start + 4))
        self.bones.lines = o3d.utility.Vector2iVector(edges)
        selected_joints = set()
        colors = np.tile([0.3, 0.7, 1.0], (25, 1))
        for finger in finger_ids:
            start = 1 if finger == 0 else 5 * finger
            colors[start:5 * finger + 5] = [1.0, 0.6, 0.1]
            selected_joints.update(range(start, 5 * finger + 5))
        self.joints.colors = o3d.utility.Vector3dVector(colors)
        self.bones.colors = o3d.utility.Vector3dVector([colors[b] for _, b in edges])
        # Mesh thickness works independently of platform-specific line-width limits.
        self.highlights = []
        for joint in sorted(selected_joints):
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0035, resolution=8)
            mesh.paint_uniform_color([1.0, 0.6, 0.1])
            self.highlights.append((mesh, np.asarray(mesh.vertices).copy(), (joint,)))
        for a, b in edges:
            if b in selected_joints:
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0025, height=1.0, resolution=8, split=1)
                mesh.paint_uniform_color([1.0, 0.6, 0.1])
                self.highlights.append((mesh, np.asarray(mesh.vertices).copy(), (a, b)))
        self.added = False

    def update(self, poses):
        """Refresh geometry from a live frame; return False when the window closes."""
        positions = np.asarray(poses)[:, :3]
        self.joints.points = o3d.utility.Vector3dVector(positions)
        self.bones.points = o3d.utility.Vector3dVector(positions)
        for mesh, template, indices in self.highlights:
            if len(indices) == 1:
                vertices = template + positions[indices[0]]
            else:
                a, b = positions[list(indices)]
                direction = b - a
                length = np.linalg.norm(direction)
                z = direction / length if length > 1e-12 else np.array([0., 0., 1.])
                reference = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
                x = np.cross(reference, z)
                x /= np.linalg.norm(x)
                rotation = np.column_stack([x, np.cross(z, x), z])
                vertices = (template * [1, 1, length]) @ rotation.T + (a + b) / 2
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
        if not self.added:
            # Fit the initial camera to the actual hand, not an empty point cloud.
            self.vis.add_geometry(self.joints)
            self.vis.add_geometry(self.bones)
            for mesh, _, _ in self.highlights:
                self.vis.add_geometry(mesh)
            self.vis.get_render_option().point_size = 4.0
            self.added = True
        else:
            self.vis.update_geometry(self.joints)
            self.vis.update_geometry(self.bones)
            for mesh, _, _ in self.highlights:
                self.vis.update_geometry(mesh)
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def add_sample(self, poses):
        """Keep recorded target-fingertip positions visible for the whole episode."""
        fingertips = [5 * finger + 4 for finger in self.finger_ids]
        if not fingertips:
            return
        points = np.asarray(poses)[fingertips, :3].copy()
        self.trajectory.points.extend(o3d.utility.Vector3dVector(points))
        self.trajectory.colors.extend(o3d.utility.Vector3dVector(
            np.tile([0.9, 0.25, 0.2], (len(points), 1))
        ))
        if not self.trajectory_added:
            self.vis.add_geometry(self.trajectory, reset_bounding_box=False)
            self.trajectory_added = True
        else:
            self.vis.update_geometry(self.trajectory)
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()

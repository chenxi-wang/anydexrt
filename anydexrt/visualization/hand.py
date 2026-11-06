"""Live robot hand rendering in the hand coordinate frame.
Author: Chenxi Wang, Ying Feng
"""
import os
import numpy as np
import open3d as o3d
import kinpy as kp
import xml.etree.ElementTree as ET
from anydexrt.geometry.poses import pose_4x4_to_7d, pose_7d_to_4x4

class HandVisualizer:
    def __init__(self, config):
        self.urdf_file = config["urdf_path"]
        self.model_chain = kp.build_chain_from_urdf(open(self.urdf_file).read().encode("utf-8"))
        self.visuals_map = self.model_chain.visuals_map()
        self.joint_num=config["joint"]["num"]
        self.finger_num = config["finger_num"]
        self.joint_mapping = {
            int(k): v for k, v in config["joint"]["joint_idx_urdf_mapping"].items()
        }
        self.fingertip_links = {
            int(k): v for k,v in config["fingertip_link"].items()
        }
        self.hand2manus= {
            int(k): v for k, v in config["joint"]["hand2manus_mapping"].items()
        }
        self.manus2hand= {
            int(k): v for k, v in config["joint"]["manus2hand_mapping"].items()
        }
        self.joint_lower_limit = np.array(config["joint"]["limit"]["lower"])
        self.joint_upper_limit = np.array(config["joint"]["limit"]["upper"])
        self.transform=np.array(config["transform"])
        self.joint_shape=config["joint"]["shape"]
        self.curr_joints = np.zeros(self.joint_num)
        self.curr_finger_pose = None
        self.model_mesh = {}
        self.link_tf = {}
        self.mesh_scales = self.load_mesh_scales()
        self._load_mesh()

    def load_mesh_scales(self):
        tree = ET.parse(self.urdf_file)
        root = tree.getroot()
        mesh_scales = {}
        for link in root.findall('link'):
            visual = link.find('visual')
            if visual is None:
                continue

            geometry = visual.find('geometry')
            if geometry is None:
                continue

            mesh = geometry.find('mesh')
            if mesh is None:
                continue

            filename = mesh.get('filename')

            scale_str = mesh.get('scale')
            if scale_str is None:
                scale = [1.0, 1.0, 1.0]
            else:
                scale = [float(x) for x in scale_str.split()]

            mesh_scales[filename] = scale

        return mesh_scales
    
    def transform_glove_to_hand(self, glove_pose):
        glove_pose = pose_7d_to_4x4(glove_pose)
        transformed_pose = glove_pose@self.transform.T
        transformed_pose = pose_4x4_to_7d(transformed_pose)
        return transformed_pose

    def convert_joint_array_to_urdf_joint(self, joints):
        assert len(joints) == self.joint_num
        urdf_joints = {}
        for i in range(self.joint_num):
            if self.joint_mapping.get(i):
                urdf_joints[self.joint_mapping[i]] = float(joints[i])
        return urdf_joints

    def convert_urdf_transform_to_finger_pose(self, transforms):
        finger_pose = []
        for i in range(self.finger_num):
            transform = transforms[self.fingertip_links[i]]
            pose = np.concatenate([transform.pos, transform.rot])
            finger_pose.append(pose)
        finger_pose = np.stack(finger_pose, axis=0)
        return finger_pose

    def _load_mesh(self):
        cur_transforms = self.model_chain.forward_kinematics(
            self.convert_joint_array_to_urdf_joint(self.curr_joints)
        )
        self.curr_finger_pose = self.convert_urdf_transform_to_finger_pose(cur_transforms)
        for link, transform in cur_transforms.items():
            self.model_mesh[link] = {}
            self.link_tf[link] = {}
            for v in self.visuals_map[link]:
                if v.geom_param is None: continue
                if not isinstance(v.geom_param, str): continue
                mesh_path = os.path.join(os.path.dirname(self.urdf_file), v.geom_param)
                mesh_scale = self.mesh_scales.get(v.geom_param, [1.0, 1.0, 1.0])
                self.model_mesh[link][v.geom_param] = o3d.io.read_triangle_mesh(mesh_path).scale(mesh_scale[0], center=(0,0,0))
                self.model_mesh[link][v.geom_param].compute_vertex_normals()
                tf = transform.matrix()@ v.offset.matrix()
                self.link_tf[link][v.geom_param] = tf
                self.model_mesh[link][v.geom_param].transform(tf)

    def update(self, joints, joint_mask=None):
        joints = joints.reshape(-1)
        if joint_mask is None:
            joint_mask = np.ones(self.joint_num, dtype=bool)
        else:
            joint_mask = joint_mask.reshape(-1).astype(bool)
        self.curr_joints[joint_mask] = joints[joint_mask]
        cur_transforms = self.model_chain.forward_kinematics(
            self.convert_joint_array_to_urdf_joint(self.curr_joints)
        )
        self.curr_finger_pose = self.convert_urdf_transform_to_finger_pose(cur_transforms)
        for link, transform in cur_transforms.items():
            for v in self.visuals_map[link]:
                if v.geom_param is None: continue
                if not isinstance(v.geom_param, str): continue
                prev_tf = self.link_tf[link][v.geom_param]
                tf = transform.matrix()@ v.offset.matrix()
                self.link_tf[link][v.geom_param] = tf
                self.model_mesh[link][v.geom_param].transform(tf @ np.linalg.inv(prev_tf))

    def render_hand(self):
        geometries = []
        for link in self.model_mesh:
            for mesh in self.model_mesh[link].values():
                geometries.append(mesh)

        return geometries

    def get_joints(self):
        return self.curr_joints.reshape(self.joint_shape).copy()
    
    def get_finger_pose(self):
        return self.curr_finger_pose.copy()

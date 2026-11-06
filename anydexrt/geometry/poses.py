"""Pose transformation and keypoint conversion.
Author: Chenxi Wang, Ying Feng
"""
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

def quat_xyzw_to_wxyz(quats):
    quats = np.concatenate([quats[..., 3:4], quats[..., 0:3]], axis=-1)
    return quats

def pose_7d_to_4x4(pose_7d):
    pose_4x4 = np.eye(4)
    pose_4x4[:3, 3] = pose_7d[:3]
    pose_4x4[:3, :3] = quat2mat(pose_7d[3:7])
    return pose_4x4

def pose_4x4_to_7d(pose_4x4):
    pose_7d = np.concatenate([pose_4x4[:3, 3], mat2quat(pose_4x4[:3, :3])])
    return pose_7d

def manus2fingertip(manus_data:np.ndarray):
    # from thumb to pinky
    fingers_data = np.zeros((5, 7), dtype=np.float32)
    fingers_data[0] = manus_data[4]
    fingers_data[1] = manus_data[9]
    fingers_data[2] = manus_data[14]
    fingers_data[3] = manus_data[19]
    fingers_data[4] = manus_data[24]
    return fingers_data

def manus2fingers(manus_data: np.ndarray):
    fingers_data = np.zeros((21, 7), dtype=np.float32)
    fingers_data[0,:]  = manus_data[0,:]
    fingers_data[1,:]  = manus_data[1,:]
    fingers_data[2,:]  = manus_data[2,:]
    fingers_data[3,:]  = manus_data[3,:]
    fingers_data[4,:]  = manus_data[4,:]
    fingers_data[5,:]  = manus_data[6,:]
    fingers_data[6,:]  = manus_data[7,:]
    fingers_data[7,:]  = manus_data[8,:]
    fingers_data[8,:]  = manus_data[9,:]
    fingers_data[9,:]  = manus_data[11,:]
    fingers_data[10,:] = manus_data[12,:]
    fingers_data[11,:] = manus_data[13,:]
    fingers_data[12,:] = manus_data[14,:]
    fingers_data[13,:] = manus_data[16,:]
    fingers_data[14,:] = manus_data[17,:]
    fingers_data[15,:] = manus_data[18,:]
    fingers_data[16,:] = manus_data[19,:]
    fingers_data[17,:] = manus_data[21,:]
    fingers_data[18,:] = manus_data[22,:]
    fingers_data[19,:] = manus_data[23,:]
    fingers_data[20,:] = manus_data[24,:]
    return fingers_data

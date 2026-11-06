"""MANUS glove interface.
Author: Chenxi Wang, Ying Feng
"""
import time
import threading
import zmq
import numpy as np
from anydexrt.geometry.poses import quat_xyzw_to_wxyz, pose_7d_to_4x4, pose_4x4_to_7d, manus2fingertip

class MANUSGlove:
    def __init__(self, glove_sn, addr="tcp://127.0.0.1:8000",hand_type="right"):
        self.glove_sn = glove_sn.lower()
        self.hand_type = hand_type
        print("Connecting to MANUS SDK ...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(addr)
        print("MANUS SDK connected")

        self.stopped = True
        self.hand_poses = None
        self._start_pub()
        if not self._wait_for_signal():
            self.stop()
            raise TimeoutError("No MANUS frames received; check serial and stream address")

    def _start_pub(self):
        self.stopped = False
        thread = threading.Thread(target=self.update_data, name="GloveDataUpdater")
        thread.daemon = True
        self.thread = thread
        thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.socket.close(linger=0)
        self.context.term()

    def _wait_for_signal(self, step_time=0.01, timeout=5):
        """Wait for the tracker signals to be available."""
        tic = time.time()
        while True:
            hand_pose = self.get_hand_poses()
            if hand_pose is not None:
                break

            if time.time() - tic > timeout:
                print(f"Failed to get glove signals in {timeout}s")
                return False

            time.sleep(step_time)

        return True

    def update_data(self, wait_time=0.01):
        while not self.stopped:
            time.sleep(wait_time)
            try:
                message = self.socket.recv()
            except zmq.Again:
                continue
            message = message.decode('utf-8')
            data = message.split(",")
            if len(data) != 176 or data[0].lower() != self.glove_sn:
                continue
            try:
                poses = self.parse_full_skeleton(data[1:176])
            except ValueError:
                continue
            self.hand_poses = poses.copy()

    def _transform_pose(self, pose, base_pose=None):
        if self.hand_type == "left":
            mirror = np.eye(4)
            mirror[0, 0] = -1
            trans = np.array([
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1],
            ])
            pose = mirror @ trans @ pose @ mirror
            rot = np.array([
                [ 1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1],
            ])
            pose = pose @ rot
        else:
            pose = np.linalg.inv(base_pose) @ pose
            trans = np.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            pose = trans @ pose
            rot = np.array([
                [ 1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1],
            ])
            pose = pose @ rot.T

        return pose

    def parse_full_skeleton(self, raw_hand_data):
        data = list(map(float, raw_hand_data))
        poses = np.array(data).reshape(25, 7)
        poses[:, 3:7] = quat_xyzw_to_wxyz(poses[:, 3:7])
        base_pose = pose_7d_to_4x4(poses[0])
        for i in range(len(poses)):
            pose = pose_7d_to_4x4(poses[i])
            pose = self._transform_pose(pose,base_pose)
            poses[i] = pose_4x4_to_7d(pose)

        return poses

    def get_hand_poses(self, format="raw"):
        assert format in ["raw", "fingertip"]
        if self.hand_poses is None:
            return None
        hand_poses = self.hand_poses.copy()
        if format == "fingertip":
            hand_poses = manus2fingertip(hand_poses)
        return hand_poses.copy()

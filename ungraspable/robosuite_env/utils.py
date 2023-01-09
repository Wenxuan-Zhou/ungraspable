"""
Misc utility functions
"""
import numpy as np

from ungraspable.robosuite_env.gym_rotations import quat_mul, quat_conjugate, euler2quat, quat2euler


def angle_diff(quat_a, quat_b):
    """
    Calculate angle difference between two quaternions.
    Following mujoco convension "wxyz".
    """
    quat_diff = quat_mul(quat_a, quat_conjugate(quat_b))
    a_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    return a_diff


def angle_diff_vec(vec1, vec2, degree=True):
    """
    Calculate angle difference between two vectors.
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    prod = np.dot(vec1, vec2)
    prod = np.clip(prod, -1, 1)
    angle = np.arccos(prod)
    if degree:
        angle = angle / np.pi * 180
    return angle


def convert_to_batch(arr):
    """
    Add a batch dimension if necessary.
    """
    if len(arr.shape) == 1:
        arr = arr[np.newaxis]
    return arr


def quat_rot_vec_arr(q, v0):
    """
    convert
    """
    q_v0 = np.array([np.zeros_like(v0[..., 0]), v0[..., 0], v0[..., 1], v0[..., 2]]).transpose()
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[..., 1:]
    return v


def get_global_pose(frame_pos, frame_quat, pos_in_frame, quat_in_frame):
    """
    Convert a pose from the local coordinate to the global coordinate.
    """
    quat = quat_mul(frame_quat, quat_in_frame)
    pos = quat_rot_vec_arr(frame_quat, pos_in_frame) + frame_pos
    return pos, quat


def get_local_pose(frame_pos, frame_quat, global_pos, global_quat):
    """
    Convert a pose from the global coordinate to the local coordinate.
    """
    local_quat = quat_mul(quat_conjugate(frame_quat), global_quat)
    local_pos = quat_rot_vec_arr(quat_conjugate(frame_quat), global_pos - frame_pos)
    return local_pos, local_quat


def clean_xzplane_pose(pos, quat, offset=False):
    """
    Project a 6D pose to the xz plane.
    """
    if offset:
        # An ugly implementation that avoids discontinuity of euler output.
        rotate_back = euler2quat(np.array([0, -np.pi / 2, 0]))
        quat = quat_mul(quat, rotate_back)
    euler = quat2euler(quat)
    return np.array([pos[0], pos[2], euler[1]])


def clean_6d_pose(pos, quat, offset=False):
    """
    Concatenate the pose
    """
    return np.concatenate([pos, quat])

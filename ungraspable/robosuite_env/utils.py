from ungraspable.robosuite_env.gym_rotations import quat_mul, quat_conjugate, euler2quat, quat2euler
import numpy as np


def angle_diff(quat_a, quat_b):
    # Subtract quaternions and extract angle between them.
    quat_diff = quat_mul(quat_a, quat_conjugate(quat_b))
    a_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    return a_diff


def convert_to_batch(arr):
    if len(arr.shape) == 1:
        arr = arr[np.newaxis]
    return arr


def quat_rot_vec_arr(q, v0):
    q_v0 = np.array([np.zeros_like(v0[...,0]), v0[...,0], v0[...,1], v0[...,2]]).transpose()
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[..., 1:]
    return v


def get_global_pose(frame_pos, frame_quat, pos_in_frame, quat_in_frame):
    # Convert the grasp pose in the object coordinate to the global coordinate
    quat = quat_mul(frame_quat, quat_in_frame)
    pos = quat_rot_vec_arr(frame_quat, pos_in_frame) + frame_pos
    return pos, quat


def get_local_pose(frame_pos, frame_quat, global_pos, global_quat):
    # Convert the grasp pose in the global coordinate to the object coordinate
    local_quat = quat_mul(quat_conjugate(frame_quat), global_quat)
    local_pos = quat_rot_vec_arr(quat_conjugate(frame_quat), global_pos - frame_pos)
    return local_pos, local_quat


def clean_xzplane_pose(pos, quat, gripper_correction=False):
    if gripper_correction:
        rotate_back = euler2quat(np.array([0, -np.pi / 2, 0]))  # rotate 180 degree
        quat = quat_mul(quat, rotate_back)
    euler = quat2euler(quat)
    return np.array([pos[0], pos[2], euler[1]])


def clean_6d_pose(pos, quat, gripper_correction=False):
    return np.concatenate([pos, quat])


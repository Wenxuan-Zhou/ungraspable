import numpy as np
from ungraspable.robosuite_env.gym_rotations import euler2quat, quat_mul
from ungraspable.robosuite_env.utils import get_global_pose


class GraspSampler(object):
    def __init__(self, env):
        self.env = env

    def sample_goal(self):
        # Sample grasp location from the edges of a square and point to the center
        # Returns position and quaternion of the grip_site in the cube coordinate
        if self.env.goal_range == "left":
            rand_num = np.random.uniform(0, 1)
        elif self.env.goal_range == "front":
            rand_num = np.random.uniform(1, 2)
        elif self.env.goal_range == "right":
            rand_num = np.random.uniform(2, 3)
        elif self.env.goal_range == "back":
            rand_num = np.random.uniform(3, 4)
        elif self.env.goal_range == "all":
            rand_num = np.random.rand()*4
        elif self.env.goal_range == "fixed":
            rand_num = 1.5  # top middle grasp point by default
        else:  # Use threshold
            rand_num = np.random.uniform(self.env.goal_range_min, self.env.goal_range_max)

        # Parameterize the grasp poses
        # \\| | | | |// form 0 to 1
        def param_to_pose(num):
            p, a = 0, 0
            if num < 0.25:  # Left corner
                # convert range from [0, 0.25) to [-np.pi/4, 0)
                p, a = -1, - np.pi * (num - 0.25)
            elif 0.25 <= num < 0.75:  # On the edge
                # convert range from [0.25, 0.75] to [-1, 1]
                p, a = (num - 0.5)*4, 0
            elif 0.75 <= num < 1:  # Right corner
                # convert range from [0.75, 1) to [0, np.pi/4)
                p, a = 1, - np.pi * (num - 0.75)
            return p, a

        # Generate a random number from 0~4 representing four edges
        rand_num = rand_num % 4
        if rand_num < 1:  # Left
            pp, aa = param_to_pose(rand_num)
            pos_x, pos_y, angle = -pp, -1, np.pi/2 + aa
        elif rand_num < 2:  # Front
            pp, aa = param_to_pose(rand_num-1)
            pos_x, pos_y, angle = -1, pp, aa
        elif rand_num < 3:  # Right
            pp, aa = param_to_pose(rand_num-2)
            pos_x, pos_y, angle = pp, 1, -np.pi/2 + aa
        else:  # Back
            pp, aa = param_to_pose(rand_num-3)
            pos_x, pos_y, angle = 1, -pp, -np.pi + aa

        # Calculate the position of the grasp based on the shape so that the grasp is on the edge
        object_shape = self.env.cube.size
        angle = angle  # + np.random.uniform(-30, 30)*np.pi/180
        pos_x = pos_x * (object_shape[0] - 0.02)
        pos_y = pos_y * (object_shape[1] - 0.02)
        pos = np.array([pos_x, pos_y, 0])
        rotation = np.array([0, 0, angle])  # Additional rotation along z axis in the object frame

        # Default orientation of the grip_site wrt global coordinate
        default_quat = euler2quat(np.array([0, np.pi/2, 0]))
        quat = quat_mul(euler2quat(rotation), default_quat)

        # Convert grip_site pose to the base of the target gripper object for visualization
        local_pos = np.array([0, 0, -0.097])
        local_quat = np.array([1, 0, 0, 0])
        pos, quat = get_global_pose(pos, quat, local_pos, local_quat)

        return np.concatenate([pos, quat])

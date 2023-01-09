import numpy as np
from rlkit.torch.core import torch_ify

from ungraspable.robosuite_env.base_env import BaseEnv
from ungraspable.robosuite_env.gym_rotations import euler2quat, quat_mul
from ungraspable.robosuite_env.utils import angle_diff, convert_to_batch, get_local_pose, get_global_pose, \
    clean_xzplane_pose, clean_6d_pose


class OccludedGraspingSimEnv(BaseEnv):
    """
    Defines the task-related properties of the simulation environment such as goals and rewards.
    """

    def __init__(
            self,
            robots,
            goal_range=None,
            goal_range_min=1.5,
            goal_range_max=1.5,
            goal_selection=None,
            alpha1=50.,
            alpha2=2.,
            beta=200.,
            **kwargs
    ):

        # Reward function weights
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

        self.goal = None

        # Set goal parameters here to be accessible in adr
        self.goal_range = goal_range
        self.goal_range_min = np.float(goal_range_min)
        self.goal_range_max = np.float(goal_range_max)
        assert self.goal_range_min <= self.goal_range_max
        assert self.goal_range_min >= 0.
        assert self.goal_range_max <= 4.

        # Grasp selection parameters
        self.goal_selection = goal_selection
        self.num_goals = 50
        self.qf = None
        self.policy = None
        self.goal_set = None

        super().__init__(
            robots=robots,
            **kwargs
        )

    """
    Observations
    """

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=True)
        """
        Observation keys: odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_pos_vel', 'robot0_eef_ori_vel', 'robot0_joint_torque', 
        'robot0_osc_desired_pos', 'robot0_osc_desired_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 
        'cube_pos', 'cube_quat', 'robot0_proprio-state', 'object-state'])
        
        These observations are treated as sensors. Derived, task-specific observations are defined here.
        """

        # Store ground truth obs for reward calculation
        # without being affected by sensor noise
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        cube_quat = np.array(self.sim.data.body_xquat[self.cube_body_id])
        obs['ground_truth_obs'] = np.concatenate([cube_pos, cube_quat])

        # Achieved grasp
        eef_pos_global = self.sim.data.body_xpos[self.sim.model.body_name2id('gripper0_right_gripper')].copy()
        eef_quat_global = self.sim.data.body_xquat[self.sim.model.body_name2id('gripper0_right_gripper')].copy()
        obs['achieved_goal'] = np.concatenate(get_local_pose(cube_pos, cube_quat, eef_pos_global, eef_quat_global))

        # Actual observation used in the policy which are derived
        # from the observables to take into account sensor noise
        controller_type = self.robots[0].controller.name
        if '3D' in controller_type:
            reduced_pose = clean_xzplane_pose
        else:
            reduced_pose = clean_6d_pose

        gripper_pose = reduced_pose(eef_pos_global, eef_quat_global, offset=False)
        cube_pose = reduced_pose(cube_pos, cube_quat, offset=True)
        local_gripper_pose = reduced_pose(obs['achieved_goal'][:3], obs['achieved_goal'][3:], offset=True)

        obs['observation'] = np.concatenate([gripper_pose, cube_pose, local_gripper_pose])

        if self.additional_obs_keys:
            for key in self.additional_obs_keys:
                obs['observation'] = np.concatenate([obs['observation'], np.array([getattr(self, key)])])

        if self.goal_selection is not None and 'pose_diff' in self.goal_selection:
            assert self.object_ori_noise == 0 and self.object_pos_noise == 0
            self.select_best_goal_by_pose(cube_pos, cube_quat, obs['achieved_goal'])
        elif self.goal_selection is not None and 'argmaxq' in self.goal_selection and self.qf is not None:
            self.select_best_goal(obs['observation'])

        # Desired grasp
        obs['desired_goal'] = self.goal.copy() if self.goal is not None else np.zeros(7)
        return obs

    """
    Reward
    """

    def _post_action(self, action):
        # Split reward and info
        reward, reward_info = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, reward_info

    def reward(self, action=None):
        # Single step reward calculation without batch dimension
        obs = self._get_observations()
        reward, info = self.compute_rewards(action, obs, verbose=True)
        assert (len(reward) == 1)
        return reward[0], {k: v[0] for k, v in info.items()}

    def compute_rewards(self, action, obs_dict, verbose=False):
        # Batch calculation of the reward. Will be called by obs_dict_replay_buffer.py in rlkit.
        # obs is the next_obs after the step.
        ground_truth_obs = convert_to_batch(obs_dict['ground_truth_obs'])
        cube_pos = ground_truth_obs[:, :3].copy()
        cube_quat = ground_truth_obs[:, 3:7].copy()
        achieved_goal = convert_to_batch(obs_dict['achieved_goal']).copy()
        desired_goal = convert_to_batch(obs_dict['desired_goal']).copy()

        # From gripper base in object frame to grip-site in local frame
        site_pos, site_quat = self.get_site_pose(desired_goal[..., :3], desired_goal[..., 3:])
        site_pos_a, site_quat_a = self.get_site_pose(achieved_goal[..., :3], achieved_goal[..., 3:])
        d_pos = np.linalg.norm(site_pos - site_pos_a, axis=-1)
        d_rot = angle_diff(site_quat, site_quat_a)
        pose_diff = -(self.alpha1 * d_pos + self.alpha2 * d_rot)

        # Target gripper penalty
        # Prepare target gripper pose in the global coordinate based on the current object pose
        gt_obs = convert_to_batch(obs_dict['ground_truth_obs'])
        cube_pos = gt_obs[..., 0:3]
        cube_quat = gt_obs[..., 3:7]
        # Convert desired grasp pose to global coordinate
        eef_pos, eef_quat = get_global_pose(cube_pos, cube_quat, desired_goal[..., :3], desired_goal[..., 3:])
        # Calculate penalty on occluded target gripper
        gripper_penalty = self.target_gripper_penalty(eef_pos, eef_quat)

        reward = pose_diff + gripper_penalty

        if verbose:
            success = (gripper_penalty == 0).astype(np.float) \
                      * (d_pos < 0.03).astype(np.float) \
                      * (d_rot / np.pi * 180 < 10).astype(np.float)
            reward_info = {'pos_diff': d_pos * 100,  # m to cm
                           'rot_diff': d_rot / np.pi * 180,  # Radian to degree
                           'flip': (gripper_penalty == 0).astype(np.float),
                           'success': success,
                           'target_finger_penalty': gripper_penalty,
                           'pos_rot_reward': pose_diff}
            return reward, reward_info
        else:
            return reward

    def get_site_pose(self, gripper_pos, gripper_quat):
        # Get global pose of the grip_site
        local_pos = np.tile(self.sim.model.site_pos[self.sim.model.site_name2id('target:grip_site')],
                            (gripper_pos.shape[0], 1))
        local_quat = np.tile(self.sim.model.site_quat[self.sim.model.site_name2id('target:grip_site')],
                             (gripper_pos.shape[0], 1))
        globel_pos, global_quat = get_global_pose(gripper_pos, gripper_quat, local_pos, local_quat)
        return globel_pos, global_quat

    def target_gripper_penalty(self, eef_pos, eef_quat):
        # Points on the gripper to decide if it is occluded or not
        sites = np.array([
            [-0.04, 0., 0.0524],
            [0.04, 0., 0.0524],
            [-0.04, 0., 0.097],
            [0.04, 0., 0.097],
            [0.09, 0., 0.],
            [-0.09, 0., 0.],
        ])

        batch_size = eef_pos.shape[0]
        table_top = self.table_offset[2]
        penalty_list = []
        for site in sites:
            pos, _ = get_global_pose(eef_pos, eef_quat,
                                     np.repeat(np.array([site]), batch_size, axis=0),
                                     np.repeat(np.array([[1., 0., 0., 0.]]), batch_size, axis=0))
            height_margin = pos[..., 2] - table_top
            penalty = np.clip(height_margin, None, 0)
            penalty_list.append(penalty)

        penalty_arr = np.vstack(penalty_list)
        total_penalty = np.mean(penalty_arr, axis=0) * self.beta
        return total_penalty

    def grasp_and_lift(self, render, video_writer):
        # Set back gripper joint damping to enable gripper action
        gripper = self.robots[0].gripper
        gripper.actuator[0].set("kp", str(1))
        gripper.actuator[1].set("kp", str(1))

        def step(action):
            # Step the robot while closing the gripper
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, True)
                for _ in range(5):
                    self.robots[0].grip_action(gripper=gripper, gripper_action=[1.])
                self.sim.step()

        controller_type = self.robots[0].controller.name
        if '3D' in controller_type or 'XZPLANE' in controller_type:
            action_sequence = np.zeros((7, 3))
            action_sequence[3:5, :] = np.array([-1, 1., 0])
        else:
            action_sequence = np.zeros((7, 6))
            action_sequence[3:5, :] = np.array([-1, 0., 1, 0., 0., 0.])

        for action in action_sequence:
            step(action)
            if video_writer is not None:
                import cv2
                # We need to directly grab full observations so we can get image data
                full_obs = self._get_observations()
                # Grab image data (assume relevant camera name is the first in the env camera array)
                img = full_obs[self.camera_names[0] + "_image"]
                img = cv2.flip(img, 0)
                # Write to video writer
                video_writer.append_data(img)

            if render:
                self.render()

        object_height = np.array(self.sim.data.body_xpos[self.cube_body_id])[2]
        success = float(object_height > 0.20)
        return success

    """
    Target grasp related functions
    """

    def sample_goal(self):
        # Sample grasp location from the edges of a square and point to the center
        # Returns position and quaternion of the grip_site in the cube coordinate
        if self.goal_range == "left":
            rand_num = np.random.uniform(0, 1)
        elif self.goal_range == "front":
            rand_num = np.random.uniform(1, 2)
        elif self.goal_range == "right":
            rand_num = np.random.uniform(2, 3)
        elif self.goal_range == "back":
            rand_num = np.random.uniform(3, 4)
        elif self.goal_range == "all":
            rand_num = np.random.rand() * 4
        elif self.goal_range == "fixed":
            rand_num = 1.5  # top middle grasp point by default
        else:  # Use threshold
            rand_num = np.random.uniform(self.goal_range_min, self.goal_range_max)

        # Parameterize the grasp poses
        # \\| | | | |// form 0 to 1
        def param_to_pose(num):
            p, a = 0, 0
            if num < 0.25:  # Left corner
                # convert range from [0, 0.25) to [-np.pi/4, 0)
                p, a = -1, - np.pi * (num - 0.25)
            elif 0.25 <= num < 0.75:  # On the edge
                # convert range from [0.25, 0.75] to [-1, 1]
                p, a = (num - 0.5) * 4, 0
            elif 0.75 <= num < 1:  # Right corner
                # convert range from [0.75, 1) to [0, np.pi/4)
                p, a = 1, - np.pi * (num - 0.75)
            return p, a

        # Generate a random number from 0~4 representing four edges
        rand_num = rand_num % 4
        if rand_num < 1:  # Left
            pp, aa = param_to_pose(rand_num)
            pos_x, pos_y, angle = -pp, -1, np.pi / 2 + aa
        elif rand_num < 2:  # Front
            pp, aa = param_to_pose(rand_num - 1)
            pos_x, pos_y, angle = -1, pp, aa
        elif rand_num < 3:  # Right
            pp, aa = param_to_pose(rand_num - 2)
            pos_x, pos_y, angle = pp, 1, -np.pi / 2 + aa
        else:  # Back
            pp, aa = param_to_pose(rand_num - 3)
            pos_x, pos_y, angle = 1, -pp, -np.pi + aa

        # Calculate the position of the grasp based on the shape so that the grasp is on the edge
        object_shape = self.cube.size
        angle = angle  # + np.random.uniform(-30, 30)*np.pi/180
        pos_x = pos_x * (object_shape[0] - 0.02)
        pos_y = pos_y * (object_shape[1] - 0.02)
        pos = np.array([pos_x, pos_y, 0])
        rotation = np.array([0, 0, angle])  # Additional rotation along z axis in the object frame

        # Default orientation of the grip_site wrt global coordinate
        default_quat = euler2quat(np.array([0, np.pi / 2, 0]))
        quat = quat_mul(euler2quat(rotation), default_quat)

        # Convert grip_site pose to the base of the target gripper object for visualization
        local_pos = np.array([0, 0, -0.097])
        local_quat = np.array([1, 0, 0, 0])
        pos, quat = get_global_pose(pos, quat, local_pos, local_quat)

        return np.concatenate([pos, quat])

    def _pre_action(self, action, policy_step=False):
        # Update target gripper pose in this function so that it is updated for each simulation timestep rather than
        # each environment timestep.
        self.render_target_gripper()
        return super()._pre_action(action, policy_step)

    def reset(self):
        super().reset()
        self.goal = self.sample_goal()
        self.goal_set = None
        self.render_target_gripper()
        return self._get_observations(force_update=True)

    def sample_goals(self, num_goals):
        # To be used in rlkit HER
        goals = [self.sample_goal() for _ in range(num_goals)]
        return {"desired_goal": np.vstack(goals)}

    def render_target_gripper(self):
        # Set the target(shadow) gripper to follow the cube for visualization purpose. The pose of the target gripper
        # is not used for observation of reward calculation.
        cube_quat = self.sim.data.body_xquat[self.cube_body_id]
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        # Convert goal to the global pose of the grip_site
        pos, quat = get_global_pose(cube_pos, cube_quat, self.goal[:3], self.goal[3:])

        self.sim.model.body_quat[self.sim.model.body_name2id('target')] = quat
        self.sim.model.body_pos[self.sim.model.body_name2id('target')] = pos
        self.sim.forward()
        return

    """
    Grasp Selection
    """

    def set_models(self, qf, policy):
        self.qf = qf
        self.policy = policy

    def select_best_goal(self, obs):
        if self.goal_set is None:
            self.goal_set = self.sample_goals(self.num_goals)['desired_goal']
        elif 'once' in self.goal_selection:
            return

        goal_set = self.goal_set.copy()
        new_obs = np.concatenate([np.tile(obs, (self.num_goals, 1)), goal_set], axis=1)
        new_obs_torch = torch_ify(new_obs)

        actions, agent_info = self.policy.get_action(new_obs_torch)
        actions_torch = torch_ify(actions)
        q_values = self.qf(new_obs_torch, actions_torch)
        new_id = np.argmax(q_values.cpu().detach().numpy())
        new_goal = goal_set[new_id]
        self.goal = new_goal
        return

    def select_best_goal_by_pose(self, cube_pos, cube_quat, achieved_goal):
        if self.goal_set is None:
            self.goal_set = self.sample_goals(self.num_goals)['desired_goal']
        elif 'once' in self.goal_selection:
            return

        desired_goal = self.goal_set
        achieved_goal = np.repeat(achieved_goal.reshape(1, -1), self.num_goals, axis=0)
        cube_pos = np.repeat(cube_pos.reshape(1, -1), self.num_goals, axis=0)
        cube_quat = np.repeat(cube_quat.reshape(1, -1), self.num_goals, axis=0)

        # From gripper base in object frame to grip-site in local frame
        site_pos, site_quat = self.get_site_pose(desired_goal[..., :3], desired_goal[..., 3:])
        site_pos_a, site_quat_a = self.get_site_pose(achieved_goal[..., :3], achieved_goal[..., 3:])
        d_pos = np.linalg.norm(site_pos - site_pos_a, axis=-1)
        d_rot = angle_diff(site_quat, site_quat_a)
        pose_diff = -(50. * d_pos + 2. * d_rot)

        new_id = np.argmax(pose_diff)
        new_goal = self.goal_set[new_id]
        self.goal = new_goal
        return

"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from robosuite.wrappers import GymWrapper


class GoalEnvWrapper(GymWrapper):
    def __init__(self, env, keys=None):
        # Run super method
        super(GymWrapper, self).__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self._flatten_obs(self.env.reset())
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ground_truth_obs=spaces.Box(-np.inf, np.inf, shape=obs['ground_truth_obs'].shape, dtype='float32'),
        ))
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict, verbose=False):
        return {
            'observation': obs_dict['observation'],
            'achieved_goal': obs_dict['achieved_goal'],
            'desired_goal': obs_dict['desired_goal'],
            'ground_truth_obs': obs_dict['ground_truth_obs'],
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert NotImplementedError

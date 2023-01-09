import cv2
import datetime
import imageio
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
from rlkit.core import logger
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch.sac.policies import MakeDeterministic

from ungraspable.rlkit_utils.rlkit_custom import get_custom_generic_path_information


def rollout_grasp_selection(env, qf, policy, args, env_args, video_writer):
    """
    Rollout with grasp selection based on the Q-function.
    """
    if hasattr(env, "set_models"):
        env.set_models(qf, policy)

    ranges = ['training', 'all']
    selections = ['uniform', 'argmaxq', 'pose_diff', 'argmaxq_once', 'pose_diff_once']
    records = []
    for training_range in ranges:
        for selection in selections:
            print(datetime.datetime.now())
            print(selection + "_" + training_range)

            # Select range of selection
            if training_range == 'training':
                # Use default range from adr training
                print(env.unwrapped.goal_range, env.unwrapped.goal_range_min, env.unwrapped.goal_range_max)
            elif training_range == 'all':
                env.unwrapped.goal_range = 'all'

            # Select selection method
            if selection == 'uniform':
                env.unwrapped.goal_selection = None
            else:
                env.unwrapped.goal_selection = selection

            # Run rollout
            eval_dict = simulate_policy(
                env=env,
                policy=policy,
                horizon=env_args["horizon"],
                render=not args.record_video and args.camera != 'none',
                video_writer=video_writer,
                num_episodes=args.num_episodes,
                printout=True,
                grasp_and_lift=args.grasp_and_lift,
            )
            eval_dict['eval_label'] = selection + "_" + training_range
            eval_dict['goal_range'] = env.unwrapped.goal_range
            eval_dict['goal_range_min'] = env.unwrapped.goal_range_min
            eval_dict['goal_range_max'] = env.unwrapped.goal_range_max
            records.append(eval_dict)


def load_adr_progress(env, data):
    """
    Load the training progress of ADR.
    """
    adr_key = None
    for k in data.keys():
        if 'adr_params' in k:
            adr_key = k
            break
    if adr_key is not None:
        for k, v in data[adr_key].items():
            val = getattr(env.env, v['name'])
            if v['finished']:
                print(v['name'], "\t default:{:.2f} \t new:{:.2f} (finished)".format(val, v['curr_val']))
                setattr(env.env, v['name'], v['curr_val'])
            elif v['curr_val'] != val and v['curr_val'] - v['inc'] != val:
                print(v['name'], "\t default:{:.2f} \t new:{:.2f}".format(val, v['curr_val'] - v['inc']))
                setattr(env.env, v['name'], v['curr_val'] - v['inc'])
            else:
                print(v['name'], "\t default:{:.2f}".format(val))

    return


def load_model(model_path,
               return_qf=False,
               printout=False,
               use_gpu=False):
    """
    Load trained model and corresponding policy.
    """

    map_location = torch.device("cuda") if use_gpu else torch.device("cpu")
    data = torch.load(model_path, map_location=map_location)
    policy = data['evaluation/policy']
    qf = data['trainer/qf1']

    if printout:
        print("Policy loaded")

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()
        qf.cuda()

    if return_qf:
        return policy, qf, data

    return policy, data


def simulate_policy(
        env,
        policy,
        horizon,
        render=False,
        video_writer=None,
        num_episodes=np.inf,
        printout=False,
        return_paths=False,
        grasp_and_lift=False):
    if printout:
        print("Simulating policy...")

    # Create var to denote how many episodes we're at
    ep = 0

    # Loop through simulation rollouts
    eval_paths = []
    while ep < num_episodes:
        if printout:
            print("Rollout episode {}".format(ep))
        path = rollout(
            env,
            policy,
            max_path_length=horizon,
            render=render,
            video_writer=video_writer,
            grasp_and_lift=grasp_and_lift,
        )
        eval_paths.append(path.copy())
        # Log diagnostics if supported by env
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

        # Increment episode count
        ep += 1

    if return_paths:
        return get_custom_generic_path_information(eval_paths), eval_paths

    return get_custom_generic_path_information(eval_paths)


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        video_writer=None,
        grasp_and_lift=False,
):
    """
    Custom rollout function that extends the basic rlkit functionality in the following ways:
    - Allows for automatic video writing if @video_writer is specified

    Added args:
        video_writer (imageio.get_writer): If specified, will write image frames to this writer

    The following is pulled directly from the rlkit rollout(...) function docstring:

    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0

    # Only render if specified AND there's no video writer
    if render and video_writer is None:
        env.render(**render_kwargs)

    # Grab image data to write to video writer if specified
    if video_writer is not None:
        # We need to directly grab full observations so we can get image data
        full_obs = env._get_observations()
        img = full_obs[env.camera_names[0] + "_image"]
        img = cv2.flip(img, 0)
        video_writer.append_data(img)

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)

        # Grab image data to write to video writer if specified
        if video_writer is not None:
            # We need to directly grab full observations so we can get image data
            full_obs = env._get_observations()

            # Grab image data (assume relevant camera name is the first in the env camera array)
            img = full_obs[env.camera_names[0] + "_image"]
            img = cv2.flip(img, 0)

            # Write to video writer
            video_writer.append_data(img)

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    if grasp_and_lift:
        lift_success = env.grasp_and_lift(render, video_writer)
        for env_info in env_infos:
            env_info['lift_success'] = 0
        env_infos[-1]['lift_success'] = lift_success

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


class FakeVideoWriter(object):
    """
    Write images into a folder instead of a video
    """

    def __init__(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.count = 0

    def append_data(self, img):
        filename = '{:03}.png'.format(self.count)
        print(img.shape)
        imageio.imwrite(os.path.join(self.dirname, filename), img[450:900, 150:600, :])
        self.count += 1

    def close(self):
        pass

"""
Utility functions for parsing / processing command line arguments
"""

import argparse

from ungraspable.rlkit_utils.her_sac import AGENTS

# Define mapping from string True / False to bool True / False
BOOL_MAP = {
    "true": True,
    "false": False
}

# Define parser
parser = argparse.ArgumentParser(description='RL args using agents / algs from rlkit and envs from robosuite')

# Add seed arg always
parser.add_argument(
    '--seed', type=int, default=0, help='random seed (default: 0)')

# Add experiment info
parser.add_argument(
    '--ExpID', type=int, default=9999, help='Experiment ID (default: 9999)')
parser.add_argument(
    '--ExpGroup', type=str, default='tmp', help='Experiment Group')


def add_og_args():
    """
    Adds args specific to OccludedGraspingSimEnv
    """
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Include adr params in obs'
    )
    parser.add_argument(
        '--goal_range',
        type=str,
        default="fixed",
        help='Mode of target grasp pose'
    )
    parser.add_argument(
        '--goal_range_min',
        type=float,
        default=1.5,
        help='Min range of target grasp pose'
    )
    parser.add_argument(
        '--goal_range_max',
        type=float,
        default=1.5,
        help='Max range of target grasp pose'
    )
    parser.add_argument(
        '--alpha1',
        type=float,
        default=50.,
        help='reward weights for position diff'
    )
    parser.add_argument(
        '--alpha2',
        type=float,
        default=2.,
        help='reward weights for ori diff'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=200.,
        help='reward weights for penalty'
    )


def add_robosuite_args():
    """
    Adds robosuite args to command line arguments list
    """
    parser.add_argument(
        '--env',
        type=str,
        default='OccludedGraspingSimEnv',
        help='Robosuite env to run test on')
    parser.add_argument(
        '--robots',
        nargs="+",
        type=str,
        default='Panda',
        help='Robot(s) to run test with')
    parser.add_argument(
        '--horizon',
        type=int,
        default=40,
        help='max num of timesteps for each episode')
    parser.add_argument(
        '--policy_freq',
        type=int,
        default=2,
        help='Policy frequency for environment (Hz)')
    parser.add_argument(
        '--controller',
        type=str,
        default="OSC_CUSTOMIZED_6D",
        help='controller to use for robot environment. Either name of controller for default config or filepath to custom'
             'controller config')


def add_agent_args():
    """
    Adds args necessary to define a general agent and trainer in rlkit
    """
    parser.add_argument(
        '--agent',
        type=str,
        default="HER-SAC",
        choices=AGENTS,
        help='Agent to use for training')
    parser.add_argument(
        '--qf_hidden_sizes',
        nargs="+",
        type=int,
        default=[512, 512, 512],
        help='Hidden sizes for Q network ')
    parser.add_argument(
        '--policy_hidden_sizes',
        nargs="+",
        type=int,
        default=[512, 512, 512],
        help='Hidden sizes for policy network ')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor')
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=1e-3,  # 3e-4,
        help='Learning rate for policy')
    parser.add_argument(
        '--qf_lr',
        type=float,
        default=5e-4,  # 3e-4,
        help='Quality function learning rate')

    # SAC-specific
    parser.add_argument(
        '--soft_target_tau',
        type=float,
        default=5e-3,
        help='Soft Target Tau value for Value function updates')
    parser.add_argument(
        '--target_update_period',
        type=int,
        default=1,
        help='Number of steps between target updates')
    parser.add_argument(
        '--no_auto_entropy_tuning',
        action='store_true',
        help='Whether to automatically tune entropy or not (default is ON)')
    parser.add_argument(
        '--target_entropy',
        type=float,
        default=None,
        help='Target entropy')
    parser.add_argument(
        '--fixed_alpha',
        type=float,
        default=1,
        help='Value for alpha if no auto entropy tuning')

    # TD3-specific
    parser.add_argument(
        '--target_policy_noise',
        type=float,
        default=0.2,
        help='Target noise for policy')
    parser.add_argument(
        '--policy_and_target_update_period',
        type=int,
        default=2,
        help='Number of steps between policy and target updates')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Tau value for training')

    # HER-specific
    parser.add_argument(
        '--rollout_goals',
        type=float,
        default=0.4,
        help='Percentage of rollout goals in HER')

    parser.add_argument(
        '--env_goals',
        type=float,
        default=0.,
        help='Percentage of random goals in HER')

    # ADR-specific
    parser.add_argument(
        '--adr_mode',
        type=str,
        default=None,
        help='Automatic Domain Randomization config file name')


def add_training_args():
    """
    Adds training parameters used during the experiment run
    """
    parser.add_argument(
        '--variant',
        type=str,
        default=None,
        help='If set, will use stored configuration from the specified filepath (should point to .json file)')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1000,
        help='Number of epochs to run')
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=20,
        help='Number of train loops per eval')
    parser.add_argument(
        '--trains_per_train_loop',
        type=int,
        default=1600,
        help='Number of training steps to take per training loop')
    parser.add_argument(
        '--expl_ep_per_train_loop',
        type=int,
        default=10,
        help='Number of exploration episodes to take per training loop')
    parser.add_argument(
        '--steps_before_training',
        type=int,
        default=1000,
        help='Number of exploration steps to take before starting training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size per training step')
    parser.add_argument(
        '--num_eval',
        type=int,
        default=20,
        help='Num eval episodes to run for each trial run')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./results/',
        help='directory to save runs')
    parser.add_argument(
        '--save_buffer',
        action='store_true',
        help='If true, saves the replay buffer')
    parser.add_argument(
        '--load_dir',
        type=str,
        default=None,
        help='path to the snapshot directory folder')
    parser.add_argument(
        '--load_buffer',
        type=str,
        default=None,
        help='path to the snapshot directory folder')
    parser.add_argument(
        '--load_buffer_size',
        type=int,
        default=None,
        help='path to the snapshot directory folder')


def add_rollout_args():
    """
    Adds rollout arguments needed for evaluating / visualizing a trained rlkit policy
    """
    parser.add_argument(
        '--load_dir',
        type=str,
        required=True,
        help='path to the snapshot directory folder')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Num rollout episodes to run')
    parser.add_argument(
        '--horizon',
        type=int,
        default=40,
        help='Horizon to use for rollouts (overrides default if specified)')
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='If true, uses GPU to process model')
    parser.add_argument(
        '--camera',
        type=str,
        default='frontview',
        help='Name of camera for visualization')
    parser.add_argument(
        '--record_video',
        action='store_true',
        help='If set, will save video of rollouts')
    parser.add_argument(
        '--record_images',
        action='store_true',
        help='If set, will save images of rollouts')
    parser.add_argument(
        '--itr',
        type=int,
        default=None,
        help='Load the model at a specific itr instead of the last model')
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        help='Rollout modes')
    parser.add_argument(
        '--goal_selection',
        type=str,
        default=None,
        help='Goal selection modes')
    parser.add_argument(
        '--save_paths',
        action='store_true',
        help='If true, store rollout paths')
    parser.add_argument(
        '--grasp_and_lift',
        action='store_true',
        help='If true, grasp the object at the end of the episode for visualization')


def get_env_kwargs(args):
    """
    Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for exploration env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        ignore_done=True,
    )

    if args.env == "OccludedGraspingSimEnv":
        env_kwargs.update(dict(adaptive=args.adaptive, goal_range=args.goal_range,
                               goal_range_min=args.goal_range_min, goal_range_max=args.goal_range_max,
                               alpha1=args.alpha1, alpha2=args.alpha2, beta=args.beta))

    return env_kwargs

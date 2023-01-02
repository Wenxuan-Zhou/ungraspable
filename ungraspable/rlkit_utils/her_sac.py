import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector, GoalConditionedPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from ungraspable.rlkit_utils.rlkit_custom import CustomTorchBatchRLAlgorithm
from rlkit.torch.her.her import HERTrainer
from rlkit.core import logger

import robosuite as suite
from robosuite.wrappers import GymWrapper
from ungraspable.rlkit_utils.gym_wrapper import GoalEnvWrapper
from ungraspable.rlkit_utils.adr import AdrPathCollector, get_obs_keys
from ungraspable.robosuite_env import get_controller_config
import json, os
import numpy as np
import torch

# Define agents available
AGENTS = {"SAC", "TD3", "HER-SAC"}


def experiment(variant, agent="SAC"):
    # Set random seed
    np.random.seed(variant["seed"])
    torch.manual_seed(variant["seed"])

    # Make sure agent is a valid choice
    assert agent in AGENTS, "Invalid agent selected. Selected: {}. Valid options: {}".format(agent, AGENTS)

    env_config = variant["environment_kwargs"]
    env_config.update(has_renderer=False,
                      has_offscreen_renderer=False,
                      use_object_obs=True,
                      use_camera_obs=False,
                      reward_shaping=True,
                      controller_configs=get_controller_config(env_config.pop("controller")))

    # Load ADR config
    adr_config = None
    if variant['adr_mode'] is not None:
        if os.path.isabs(variant['adr_mode']):
            custom_fpath = variant['adr_mode']
        else:
            custom_fpath = os.path.join(os.path.dirname(__file__), 'adr_config/{}.json'.format(variant['adr_mode']))
        with open(custom_fpath) as f:
            adr_config = json.load(f)
        additional_obs_keys = get_obs_keys(adr_config)
        env_config.update(additional_obs_keys=additional_obs_keys)

    # Create gym-compatible envs
    if "HER" in variant['algorithm']:
        expl_env = NormalizedBoxEnv(GoalEnvWrapper(suite.make(**env_config)))
        eval_env = NormalizedBoxEnv(GoalEnvWrapper(suite.make(**env_config)))
        obs_dim = eval_env.observation_space.spaces['observation'].low.size +\
            eval_env.observation_space.spaces['desired_goal'].low.size
    else:
        expl_env = NormalizedBoxEnv(GymWrapper(suite.make(**env_config)))
        eval_env = NormalizedBoxEnv(GymWrapper(suite.make(**env_config)))
        obs_dim = expl_env.observation_space.low.size

    action_dim = eval_env.action_space.low.size

    logger.log(f"Obs dim: {obs_dim} Action dim: {action_dim}.")

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    if "SAC" in agent:
        expl_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
        )
        eval_policy = MakeDeterministic(expl_policy)
        trainer = SACTrainer(
            env=eval_env,
            policy=expl_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
    elif "TD3" in agent:
        eval_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        es = GaussianStrategy(
            action_space=expl_env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=eval_policy,
        )
        trainer = TD3Trainer(
            policy=eval_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['trainer_kwargs']
        )
    else:
        print("Error: No valid agent chosen!")

    if variant['load_dir'] is not None:
        data = torch.load(variant['load_dir'], map_location=ptu.device)
        trainer_data = dict()
        for k in data.keys():
            if 'trainer' in k:
                trainer_data[k[8:]] = data[k]
        trainer.load_snapshot(trainer_data)
        logger.log(f"Loaded sac networks and optimizers from: " + variant['load_dir'])
        expl_policy = trainer.policy
        eval_policy = MakeDeterministic(expl_policy)
        if 'exploration/adr_params' in data.keys():
            logger.log("Load previous ADR progress:")
            adr_params = data['exploration/adr_params']
            for k, v in adr_params.items():
                default_val = getattr(expl_env.env, v['name'])
                if v['finished']:
                    logger.log("{}\t default:{:.3f} \t new:{:.3f} (finished)".format(v['name'], default_val, v['curr_val']))
                    setattr(eval_env.env, v['name'], v['curr_val'])
                    setattr(expl_env.env, v['name'], v['curr_val'])
                elif v['curr_val'] != default_val and v['curr_val'] - v['inc'] != default_val:
                    logger.log("{} \t default:{:.3f} \t new:{:.3f}".format(v['name'], default_val, v['curr_val'] - v['inc']))
                    setattr(eval_env.env, v['name'], v['curr_val'] - v['inc'])
                    setattr(expl_env.env, v['name'], v['curr_val'] - v['inc'])
                else:
                    logger.log("{} \t default:{:.3f}".format(v['name'], default_val))

    if "HER" in variant['algorithm']:
        trainer = HERTrainer(trainer)
        replay_buffer = ObsDictRelabelingBuffer(
            env=eval_env,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            internal_keys=['ground_truth_obs'],
            **variant['replay_buffer_kwargs']
        )
        if variant['load_buffer'] is not None:
            replay_buffer.load(variant['load_buffer'], variant['load_buffer_size'])
            logger.log(f"Loaded replay buffer with size {replay_buffer.num_steps_can_sample()} from: "
                       + variant['load_buffer'])

        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            eval_policy,
            observation_key='observation',
            desired_goal_key='desired_goal',
        )
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            expl_policy,
            observation_key='observation',
            desired_goal_key='desired_goal',
        )
    else:
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_kwargs']['max_size'],
            expl_env,
        )
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )

    if adr_config is not None:
        # Make sure to use env.env!
        expl_path_collector = AdrPathCollector(expl_path_collector, expl_env=expl_env.env, eval_env=eval_env.env,
                                               **adr_config)

    # Define algorithm
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()

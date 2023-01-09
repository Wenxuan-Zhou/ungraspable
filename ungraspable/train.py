import json
import os
import rlkit.torch.pytorch_util as ptu
import time
import torch
from rlkit.launchers.launcher_util import setup_logger

from ungraspable.arguments import *
from ungraspable.rlkit_utils.her_sac import experiment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def args_to_variant():
    args = parser.parse_args()

    # Construct variant to train
    if args.variant is None:
        # Convert args to variant
        trainer_kwargs = None
        if "SAC" in args.agent:
            trainer_kwargs = dict(
                discount=args.gamma,
                soft_target_tau=args.soft_target_tau,
                target_update_period=args.target_update_period,
                policy_lr=args.policy_lr,
                qf_lr=args.qf_lr,
                use_automatic_entropy_tuning=(not args.no_auto_entropy_tuning),
                target_entropy=args.target_entropy,
                fixed_alpha=args.fixed_alpha,
            )
        elif "TD3" in args.agent:
            trainer_kwargs = dict(
                target_policy_noise=args.target_policy_noise,
                policy_learning_rate=args.policy_lr,
                qf_learning_rate=args.qf_lr,
                policy_and_target_update_period=args.policy_and_target_update_period,
                tau=args.tau,
            )
        else:
            pass
        variant = dict(
            ExpID=args.ExpID,
            ExpGroup=args.ExpGroup,
            algorithm=args.agent,
            adr_mode=args.adr_mode,
            seed=args.seed,
            load_dir=args.load_dir,
            load_buffer=args.load_buffer,
            load_buffer_size=args.load_buffer_size,
            version="normal",
            replay_buffer_kwargs=dict(
                max_size=int(1E6),
                fraction_goals_rollout_goals=args.rollout_goals,  # equal to k = 4 in HER paper
                fraction_goals_env_goals=args.env_goals,
                save_buffer=args.save_buffer,
            ),
            qf_kwargs=dict(
                hidden_sizes=args.qf_hidden_sizes,
            ),
            policy_kwargs=dict(
                hidden_sizes=args.policy_hidden_sizes,
            ),
            algorithm_kwargs=dict(
                num_epochs=args.n_epochs,
                num_train_loops_per_epoch=args.eval_freq,
                num_eval_steps_per_epoch=args.horizon * args.num_eval,
                num_trains_per_train_loop=args.trains_per_train_loop,
                num_expl_steps_per_train_loop=args.horizon * args.expl_ep_per_train_loop,
                min_num_steps_before_training=args.steps_before_training,
                max_path_length=args.horizon,
                batch_size=args.batch_size,
            ),
            trainer_kwargs=trainer_kwargs,
            environment_kwargs=get_env_kwargs(args),
        )
    else:
        # Attempt to load the json file
        try:
            with open(args.variant) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening specified variant json at: {}. "
                  "Please check filepath and try again.".format(variant))

    return variant, args.log_dir


def get_logger(variant, log_folder):
    # Setup logger
    folder_name = f'Exp{variant["ExpID"]:04d}_{variant["environment_kwargs"]["env_name"]}_' + \
                  f'{variant["ExpGroup"]}-{variant["seed"]}'
    log_dir = os.path.join(log_folder, time.strftime("%m-%d") + "-" + variant["ExpGroup"], folder_name)
    save_freq = int(100 / variant["algorithm_kwargs"]["num_train_loops_per_epoch"])
    setup_logger(variant=variant, log_dir=log_dir, snapshot_gap=save_freq, snapshot_mode='gap_and_last')
    ptu.set_gpu_mode(torch.cuda.is_available())
    print(f"CUDA is_available: ", torch.cuda.is_available())


if __name__ == '__main__':
    # Add necessary command line args
    add_robosuite_args()
    add_agent_args()
    add_training_args()
    add_og_args()
    exp_variant, folder = args_to_variant()
    get_logger(exp_variant, folder)
    experiment(exp_variant, agent=exp_variant["algorithm"])

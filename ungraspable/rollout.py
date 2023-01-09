import imageio
import json
import numpy as np
import os
import pickle
import torch
from robosuite.wrappers import GymWrapper
from signal import signal, SIGINT
from sys import exit

import robosuite as suite
from ungraspable.arguments import add_rollout_args, parser
from ungraspable.rlkit_utils.rollout_utils import rollout_grasp_selection, simulate_policy, load_model, \
    load_adr_progress, FakeVideoWriter
from ungraspable.robosuite_env import get_controller_config

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Add and parse arguments
add_rollout_args()
args = parser.parse_args()

# Define callbacks
video_writer = None


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)


# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

if __name__ == "__main__":
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get path to saved model
    kwargs_fpath = os.path.join(args.load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            kwargs = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(kwargs_fpath))

    # Grab / modify env args
    env_args = kwargs["eval_environment_kwargs"] if "eval_environment_kwargs" in kwargs.keys() \
        else kwargs["environment_kwargs"]

    # Env Naming Backward compatibility
    if env_args["env_name"] == "DexEnv":
        env_args["env_name"] = "OccludedGraspingSimEnv"

    if args.horizon is not None:
        env_args["horizon"] = args.horizon
    env_args["render_camera"] = args.camera
    env_args["hard_reset"] = True
    env_args["ignore_done"] = True
    env_args["goal_selection"] = args.goal_selection

    # Specify camera name if we're recording a video
    if args.record_video:
        env_args["camera_names"] = args.camera
        env_args["camera_heights"] = 1024
        env_args["camera_widths"] = 1024

    args.load_dir = args.load_dir[:-1] if args.load_dir[-1] == '/' else args.load_dir

    # Setup video recorder if necesssary
    if args.record_video:
        # Grab name of this rollout combo
        # video_name = f'Exp{kwargs["ExpID"]:04d}-{kwargs["seed"]}-{env_args["env_name"]}-{kwargs["ExpGroup"]}'
        video_name = os.path.basename(args.load_dir)
        if args.itr is not None:
            video_name += "_itr_{}".format(args.itr)
        if args.goal_selection is not None:
            video_name += "_{}".format(args.goal_selection)
        # Calculate appropriate fps
        fps = int(env_args["control_freq"])
        # Define video writer
        foldername = os.path.dirname(args.load_dir)
        if not args.record_images:
            video_writer = imageio.get_writer(os.path.join(foldername, video_name + '.mp4'), fps=fps)
        else:
            video_writer = FakeVideoWriter(os.path.join(foldername, video_name + '_images'))

    # Pop the controller
    controller = env_args.pop("controller")
    controller_config = get_controller_config(controller)

    # Create env
    # env_args["env_name"] = "DexMultiGraspAcronym"
    key_list = [k for k in env_args.keys()]
    for k in key_list:
        if k in ['cube_size', 'cube_thickness', 'rho', 'rot']:
            print("Removing " + k + " for backward compatibility:", env_args[k])
            env_args.pop(k)
    env_suite = suite.make(**env_args,
                           controller_configs=controller_config,
                           has_renderer=not args.record_video and args.camera != 'none',
                           has_offscreen_renderer=args.record_video,
                           use_object_obs=True,
                           use_camera_obs=args.record_video,
                           reward_shaping=True
                           )

    if "HER" in kwargs["algorithm"]:
        env = GymWrapper(env_suite, keys=['observation', 'desired_goal'])
    else:
        env = GymWrapper(env_suite)

    model_filename = "params.pkl" if args.itr is None else "itr_{}.pkl".format(args.itr)

    policy, qf, data = load_model(model_path=os.path.join(args.load_dir, model_filename),
                                  return_qf=True,
                                  printout=True,
                                  use_gpu=args.gpu)

    if hasattr(env, "set_models"):
        env.set_models(qf, policy)

    load_adr_progress(env, data)

    if args.mode == 'grasp_selection':
        rollout_grasp_selection(env, qf, policy, args, env_args, video_writer)
    else:
        # Run rollout
        eval_dict, paths = simulate_policy(
            env=env,
            policy=policy,
            horizon=env_args["horizon"],
            render=not args.record_video and args.camera != 'none',
            video_writer=video_writer,
            num_episodes=args.num_episodes,
            printout=True,
            return_paths=True,
            grasp_and_lift=args.grasp_and_lift,
        )

        if args.save_paths:
            pickle.dump(paths, open(os.path.basename(args.load_dir) + ".pkl", "wb"))

        for k in eval_dict.keys():
            if k == 'FinalReward Max' or k == 'FinalReward Mean' or ('final' in k and 'Mean' in k):
                print(f"{k}:\t {eval_dict[k]:.2f}")

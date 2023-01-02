import os
from ungraspable.robosuite_env.og_env import OccludedGraspingSimEnv
from robosuite.environments.base import register_env
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config

register_env(OccludedGraspingSimEnv)


def get_controller_config(controller):
    controller_config = None

    custom_fpath = os.path.join(os.path.dirname(__file__), './controller_config/{}.json'.format(controller.lower()))
    if os.path.exists(custom_fpath):
        # Try to load from robosuite_env configs
        controller_config = load_controller_config(custom_fpath=custom_fpath)
    elif os.path.exists(controller):
        # Load from a full path
        controller_config = load_controller_config(custom_fpath=controller)
    elif controller in set(ALL_CONTROLLERS):
        # Robosuite default controllers
        controller_config = load_controller_config(default_controller=controller)

    assert controller_config is not None, f"Cannot load controller {controller}."

    return controller_config

# robosuite_env

### og_env.py
- Defines the Occluded Grasping Task including obs, reward and goals.

### base_env.py
- Configures the simulation environment of the Occluded Grasping Task
which loads the object model, the bin and the robot
- Modified from robosuite.manipulation.lift

### single_arm.py
- Loads the robot model and the controller
- Includes all the robot related obs
- Modified from robosuite.robots.single_arm

### osc.py
- Defines the controller of the robot
- 'eef' means finger tip (defined in single_arm.py)

### bin_arena.py
- Defines the robot workspace that contains one bin

### gym_rotations.py
- From gym.envs.robotics.rotations
- Copied the file over here because the newer gym version does not have gym.envs.robotics anymore.
- Note that Mujoco quaternion convention is wxyz while robosuite conversion is xyzw.

### utils.py
- Miscellaneous functions
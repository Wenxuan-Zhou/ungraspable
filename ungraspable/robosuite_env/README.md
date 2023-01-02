# DexEnv

### TODO
- Remove all dependencies of gym.envs.robotics.rotations
- Remove all usages of euler angles because the

---
### dex_env.py
- Defines the task of dexterous grasping
- Observations:
  - Object  pos,quat in global frame
  - Grip_site pos, quat in global frame
  - Grip_site pos, quat in object frame (as achieved goal)
  - Desired goal
- Reward definition

### grasp_samplers.py
- Defines how the target grasp is sampled

### base_env.py
- Defines the object model
- Loads the arena and the robot
- Includes all the object related obs

### single_arm.py
- Loads the robot model and the controller
- Includes all the robot related obs
- Be careful about end-effector. It is defined to be robot0_right_gripper 
which is not consistent with the actual orientation of the gripper base.

### osc_xzplane.py
- Defines the controller of the robot
- 'eef' means finger tip (defined in single_arm.py)

### bin_arena.py
- Defines the robot workspace that contains one bin

# Rotations
- Following Mujoco quaternion convention: wxyz (instead of robosuite conversion xyzw).
# Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity

[//]: # (<div align="center">)

[//]: # (<font size=7>**Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity**</font>)

[Wenxuan Zhou](https://wenxuan-zhou.github.io/), [David Held](https://davheld.github.io/)

Robotics Institute, Carnegie Mellon University

Conference on Robot Learning (CoRL) 2022 (Oral)

[Paper](https://arxiv.org/abs/2211.01500)
| [Website](https://sites.google.com/view/grasp-ungraspable)
| [Real robot code](https://github.com/Wenxuan-Zhou/frankapy_env)

![intro.gif](imgs/intro.gif)

[//]: # (</div>)

In this paper, we build a system based on reinforcement learning that shows 
emergent extrinsic dexterity behavior with a simple gripper 
for the "Occluded Grasping" task. This repository contains the code for the
simulation environment of the Occluded Grasping task and RL
training and rollouts. The code for the real robot can be found in 
[a separate repository](https://github.com/Wenxuan-Zhou/frankapy_env).

This repository is built on top of [robosuite-benchmark](https://github.com/ARISE-Initiative/robosuite-benchmark). The simulation environment is based on [robosuite](https://robosuite.ai/) and the RL training related code 
is based on [rlkit](https://github.com/rail-berkeley/rlkit). As an overview of this repository, [ungraspable/robosuite_env](ungraspable%2Frobosuite_env)
defines the Occluded Grasping task. [ungraspable/rlkit_utils](ungraspable/rlkit_utils) defines helper functions to be used with rlkit.

Please feel free to contact us if you have any questions on the code or anything else related to our paper!

## Installation

Clone the current repository:
```bash
$ git clone --recursive https://github.com/Wenxuan-Zhou/dexterity.git
$ cd dexterity
```

Create a conda environment with required packages.
IMPORTANT: We require the exact version of robosuite and rlkit included in this directory (included in the following yml file).
```bash
conda env create -f env.yml
source activate ungraspable
```

Use [viskit](https://github.com/vitchyr/viskit) to visualize training log files. Do not install it in the above conda environment because there are compatibility issues.

## Usage
### Training
```bash
python ungraspable/train.py --ExpID 0000
```
The results will be saved under "./results" by default. During training, you can visualize current logged runs using [viskit](https://github.com/vitchyr/viskit).

To train the policy with a multi-grasp curriculum:
```bash
python ungraspable/train.py --adr_mode 0001_ADR_MultiGrasp --ExpID 0001
```

To train the policy with Automatic Domain Randomization over physical parameters:
```bash
python ungraspable/train.py --adr_mode 0002_ADR_physics --ExpID 0002
```


### Visualizing Rollouts
To visualize a trained policy with onscreen mujoco renderer:
```bash
python ungraspable/rollout.py --load_dir results/examples/Exp0630_DexEnv_MultiGrasp-0 --camera sideview --grasp_and_lift
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{zhou2022ungraspable,
  title={Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity},
  author={Zhou, Wenxuan and Held, David},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```

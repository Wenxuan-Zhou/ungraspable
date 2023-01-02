# Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity

Code for "Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity".
The code for the real robot experiments can be found here: https://github.com/Wenxuan-Zhou/frankapy_env.

Build on top of https://github.com/ARISE-Initiative/robosuite-benchmark.

## Updates
[01.01.2023] Initial commits.

## Installation

Clone the current repository:
```bash
$ git clone https://github.com/Wenxuan-Zhou/dexterity.git
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
python ungraspable/train.py
```
The results will be saved under "./results" by default. During training, you can visualize current logged runs using [viskit](https://github.com/vitchyr/viskit).

### Visualizing Rollouts
To visualize a trained policy with onscreen mujoco renderer:
```bash
python ungraspable/rollout.py --load_dir results/examples/Exp0630_DexEnv_MultiGrasp-0 --camera sideview --grasp_and_lift
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{zhou2022ungraspable,
  title={{Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity}},
  author={Zhou, Wenxuan and Held, David},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```

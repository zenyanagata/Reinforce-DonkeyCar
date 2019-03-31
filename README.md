# Smoother and Faster

Using reeinforcement learning algorithm **PPO** and a Variational AutoEncoder (**VAE**) in the DonkeyCar simulator.

Video: [https://www.youtube.com/watch?v=udPiZA4nsK0](https://www.youtube.com/watch?v=iiuKh0yDyKE)


![result](content/sample.gif)


## Quick Start

0. Download simulator [here](https://drive.google.com/open?id=1h2VfpGHlZetL5RAPZ79bhDRkvlfuB4Wb) or build it from [source](https://github.com/tawnkramer/sdsandbox/tree/donkey)
1. Install dependencies (cf requirements.txt)
2. (optional but recommended) Download pre-trained VAE: [VAE Level 0](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) 
3. Train a control policy for 200k steps using PPO

```
python ppo_train.py -vae path-to-vae.pkl  -tb path-to-save-tensorboard-log -n 200000
```

4. Enjoy trained agent for 10k steps

```
python enjoy.py --algo ppo2 --log_path path-where-the-trained-pkl-is-saved
```

All the hyperparameters of PPO are stored in `hyperparams/ppo2.yml` and
all the congirugrations are stored in  `config.py`.

## Train the Variational AutoEncoder (VAE)

0. Collect images using the teleoperation mode:

```
python -m teleop.teleop_client --record-folder path-to-record/folder/
```

1. Train a VAE:
```
python -m vae.train --n-epochs 50 --verbose 0 --z-size 64 -f path-to-record/folder/
```

## Reproducing Results

To reproduce the results shown in the video, you have to use the default values in `config.py` and `hyperparams/ppo2.yml`.


`config.py`:

```python
MAX_STEERING_DIFF = 0.15 # 0.1 for very smooth control, but it requires more steps
MAX_THROTTLE = 0.6 # MAX_THROTTLE = 0.5 is fine, but we can go faster
MAX_CTE_ERROR = 2.0 # only used in normal mode, set it to 10.0 when 
```

`hyperparams/ppo2.yml`
```python
DonkeyVae-v0:
  # normalize: "{'norm_obs': True, 'norm_reward': False}"
  n_timesteps: !!float 10000
  policy: 'MlpPolicy'
  # n_steps min
  n_steps: 256
  noptepochs: 5
  nminibatches: 4
  ent_coef: 0.0
  learning_rate: !!float 3e-3
```

## Citing the Project

To cite this repository in publications:

```
@misc{Reinforce-DonkeyCar,
  author = {Zenya Nagata},
  title = {Smoother and Faster},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zenyanagata/Reinforce-DonkeyCar/}},
}
```

## Credits
- [Antonin Raffin](https://github.com/araffin/learning-to-drive-in-5-minutes) for the original implementation
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.
- [RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) for training/enjoy scripts.
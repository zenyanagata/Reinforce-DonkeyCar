# learning to drive in 5 minuites
# train.pyの必要最小限のコードをピックアップ
import argparse
import os
from collections import OrderedDict

import numpy as np
import yaml
from stable_baselines.common import set_global_seeds # なにこれ
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from utils.utils import make_env, ALGOS, load_vae, create_callback, get_latest_run_id

from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP,\
    MAX_CTE_ERROR, SIM_PARAMS, N_COMMAND_HISTORY, Z_SIZE, BASE_ENV, ENV_ID, MAX_STEERING_DIFF

# load_vae -> vae.z_size, normalize
# hyperparams["frame_stack"], hyperparams["policy"], hyperparams["normalize"]

parser = argparse.ArgumentParser(description="PPO")
parser.add_argument('-tb', '--tensorboard-log', type=str, default='', help='Tensorboard log dir')
parser.add_argument("--algo", type=str, help="RL Algorithm", default="ppo2",
                    required=False, choices=list(ALGOS.keys()))
parser.add_argument("--seed", type=int, default=0, help="Random generator seed")
parser.add_argument("--trained_agent", type=str, default="")
parser.add_argument("--log_interval", type=int, default=-1)
parser.add_argument("-f", "--log_folder", type=str, default="logs", help="Log folder")
parser.add_argument('-vae', "--vae_path", type=str, default="", help='Path to saved VAE')
parser.add_argument("--save_vae", action="store_true", default=False, help="Save VAE")
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                    type=int)
args = parser.parse_args()

#なにこれ
set_global_seeds(args.seed)

if args.trained_agent != "":
    assert args.trained_agent.endswith(".pkl") and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

################### Create log path #####################
tensorboard_log = None if args.tensorboard_log == "" else args.tensorboard_log + "/" + ENV_ID
log_path = os.path.join(args.log_folder, args.algo)
save_path = os.path.join(log_path, "{}_{}".format(ENV_ID, get_latest_run_id(log_path, ENV_ID) + 1))
params_path = os.path.join(save_path, ENV_ID)
os.makedirs(save_path, exist_ok=True)
os.makedirs(params_path, exist_ok=True)
#########################################################


print("="*10, ENV_ID, args.algo, "="*10)

##################### Set VAE ###########################
vae = None
if args.vae_path != "":
    print("Loading VAE ...")
    vae = load_vae(args.vae_path)
elif args.random_features:
    print("Using randomly initialized VAE")
    vae = load_vae(z_size=Z_SIZE)
    # Save network
    args.save_vae = True
else:
    print("Learning from pixels, not using VAE")
#########################################################


##################### hyperparams #######################
# Load hyperparams from yaml file
with open("hyperparams/{}.yml".format(args.algo), "r") as f:
    hyperparams = yaml.load(f)[BASE_ENV]

# 最後にsaveするためにhyperparamsをsortする
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
# vaeがsaveされるpathを追加
saved_hyperparams["vae_path"] = args.vae_path
if vae is not None:
    saved_hyperparams["z_size"] = vae.z_size

# Save simulation params
for key in SIM_PARAMS:
    saved_hyperparams[key] = eval(key)
print(saved_hyperparams)
#########################################################


################# number of timesteps ###################
# hyperparamsからn_timestepsを読み込む
# parserで指定されたら上書き
if args.n_timesteps > 0:
    n_timesteps = args.n_timesteps
else:
    n_timesteps = int(hyperparams["n_timesteps"])
del hyperparams["n_timesteps"]
#########################################################


env = DummyVecEnv([make_env(args.seed, vae=vae)])


##################### normalize #########################
# If normalize exsists in hyperparams, add in normalize_kwargs
# and set normalize to True
normalize = False
normalize_kwargs = {}
if "normalize" in hyperparams.keys():
    normalize = hyperparams["normalize"]
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams["normalize"]

# Normalize the input image
if normalize:
        print("Normalizing input and return")
        env = VecNormalize(env, **normalize_kwargs)
#########################################################


################ Optional Frame-stacking ################
n_stack = 1
if hyperparams.get("frame_stack", False):
    n_stack = hyperparams["frame_stack"]
    env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
    del hyperparams["frame_stack"]
else:
    env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
#########################################################


###################### set model ########################
# Continue training an agent 
if args.trained_agent.endswith(".pkl") and os.path.isfile(args.trained_agent):
    print("Loading pretrained agent")
    #policy should not be changed
    del hyperparams["policy"]

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tensorboard_log, full_tensorboard_log=False, verbose=1, **hyperparams)

    #なにこれ　running_averageとは
    exp_folder = args.trained_agent.split(".pkl")[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)

# Train an agent form scratch
else:
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, full_tensorboard_log=False, verbose=1, **hyperparams)
#########################################################


#################### model.learn ########################
kwargs = {}
if args.log_interval > -1:
    kwargs = {"log_interval": args.log_interval}
# kwargs.update({"callback": create_callback(args.algo,
#                                            os.path.join(save_path, ENV_ID+"_best"),
#                                            verbose=1)})

model.learn(n_timesteps, **kwargs)
#########################################################



env.reset()
if isinstance(env, VecFrameStack):
    env = env.venv
#HACK to bypass Monitor wrapper
env.envs[0].env.exit_scene()

# Save trained model
model.save(os.path.join(save_path, ENV_ID))
# Save hyperparameters
with open(os.path.join(params_path, "config.yml"), "w") as f:
    yaml.dump(saved_hyperparams, f)

if args.save_vae and vae is not None:
    print("Saving VAE")
    vae.save(os.path.join(params_path, "vae"))

if normalize:
    # Unwrap
    if isinstance(env, VecFrameStack):
        env = env.venv
    # save the running average. 
    # For testing the agent we need that normalization
    env.save_running_average(params_path)

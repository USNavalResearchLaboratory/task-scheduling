# %%
import pickle
from functools import partial
from itertools import product
from operator import attrgetter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPSpawnStrategy, DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from torch import nn, optim

from task_scheduling.algorithms import earliest_release, mcts, priority_sorter, random_sequencer
from task_scheduling.base import get_now
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.base import RandomAgent
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.features import encode_discrete_features, param_features
from task_scheduling.mdp.modules import MultiNet, VaryCNN, build_mlp, valid_logits
from task_scheduling.mdp.reinforcement import (
    MultiExtractor,
    StableBaselinesScheduler,
    ValidActorCriticPolicy,
    ValidDQNPolicy,
)
from task_scheduling.mdp.supervised import LitScheduler, TorchScheduler
from task_scheduling.results import evaluate_algorithms_gen, evaluate_algorithms_train

# from math import factorial


np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
plt.style.use("images/style.mplstyle")
plt.rc("text", usetex=False)

now = get_now()

seed = None
# seed = 12345

if seed is not None:
    seed_everything(seed)


# %% Define scheduling problem and algorithms

data_path = Path("data/")

data_tensors = data_path / "temp/tensors_1e5"

# dataset = "continuous_linear_drop_c1t8"
dataset = "temp/continuous_linear_drop_c1t8_1e5"
problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)

save_dir = "users/main_temp/"
if isinstance(problem_gen, problem_gens.Dataset):
    save_dir += f"{dataset}/"
else:
    save_dir += "other/"


# Algorithms

time_shift = True
# time_shift = False
masking = True
# masking = False

features = param_features(problem_gen.task_gen, time_shift, masking)
# features = features[1:]  # remove duration from `param_features` for radar
# features = encode_discrete_features(problem_gen)

env_params = dict(
    features=features,
    normalize=True,
    # normalize=False,
    # sort_func=None,
    sort_func="t_release",
    # sort_func=lambda task: -task.l_drop,
    time_shift=time_shift,
    masking=masking,
)

env = Index(problem_gen, **env_params)


batch_size = 200
learn_params_torch = {
    "batch_size_train": batch_size,
    "frac_val": 1 / 3,
    "batch_size_val": batch_size,
    "max_epochs": 5000,
    "shuffle": True,
    "dl_kwargs": dict(
        # num_workers=0,
        # persistent_workers=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    ),
}

model_kwargs = dict(
    optim_cls=optim.Adam,
    optim_params={"lr": 1e-4},
)

# module = MultiNet.mlp(env, hidden_sizes_ch=[], hidden_sizes_tasks=[], hidden_sizes_joint=[400])
module = MultiNet.mlp(
    env, hidden_sizes_ch=[], hidden_sizes_tasks=[], hidden_sizes_joint=[1000, 800, 500, 200]
)
# module = MultiNet.cnn(env, hidden_sizes_ch=[], hidden_sizes_tasks=[400], kernel_sizes=2,
#                       cnn_kwargs=dict(pooling_layers=[nn.AdaptiveMaxPool1d(1)]), hidden_sizes_joint=[])
# module = VaryCNN(env, kernel_len=2)

trainer_kwargs = dict(
    logger=TensorBoardLogger(save_dir + "logs/lit/", name=now),
    enable_checkpointing=False,
    log_every_n_steps=30,
    callbacks=EarlyStopping("val_loss", min_delta=1e-3, patience=200),
    default_root_dir=save_dir + "logs/lit/",
    accelerator="auto",
    # strategy=DDPStrategy(find_unused_parameters=False),
    # strategy=DDPSpawnStrategy(find_unused_parameters=False),
)


with open(data_tensors, "rb") as f:
    load_dict = pickle.load(f)
    obs, act = load_dict["obs"], load_dict["act"]


if __name__ == "__main__":
    torch_scheduler = TorchScheduler(env, module, **model_kwargs, learn_params=learn_params_torch)
    lit_scheduler = LitScheduler.from_module(
        env,
        module,
        model_kwargs,
        trainer_kwargs=trainer_kwargs,
        learn_params=learn_params_torch,
    )

    # torch_scheduler.train(obs, act, verbose=1)
    lit_scheduler.train(obs, act, verbose=1)

    plt.show()

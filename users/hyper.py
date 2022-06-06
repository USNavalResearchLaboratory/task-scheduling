import pickle
from functools import partial
from itertools import product
from operator import attrgetter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from matplotlib import pyplot as plt
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPSpawnStrategy, DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn, optim

from task_scheduling.algorithms import earliest_release, mcts, priority_sorter, random_sequencer
from task_scheduling.base import get_now
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.features import encode_discrete_features, param_features
from task_scheduling.mdp.modules import MultiNet, VaryCNN, build_mlp, valid_logits
from task_scheduling.mdp.supervised import LitScheduler, TorchScheduler
from task_scheduling.results import evaluate_algorithms_gen, evaluate_algorithms_train

# from math import factorial


np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
plt.style.use("images/style.mplstyle")
plt.rc("text", usetex=False)

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)

now = get_now()


# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.continuous_linear_drop(n_tasks=8, n_ch=1, ch_avail_lim=(0., 0.), rng=seed)
# problem_gen = problem_gens.Random.radar(n_tasks=8, n_ch=1, mode='track', rng=seed)
# problem_gen = problem_gens.Random.discrete_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path("data/")

data_tensors = data_path / "temp/tensors_1e5"


# dataset = "continuous_linear_drop_c1t8"
dataset = "temp/continuous_linear_drop_c1t8_1e5"
problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)

temp_path = "users/main_temp/optuna/"
if isinstance(problem_gen, problem_gens.Dataset):
    temp_path += f"{dataset}/"
else:
    temp_path += "other/"


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

# obs, act, *__ = env.opt_rollouts(problem_gen.n_problems, verbose=1, rng=seed)
with open(data_tensors, "rb") as f:
    load_dict = pickle.load(f)
    obs, act = load_dict["obs"], load_dict["act"]


def objective(trial):
    # batch_size = 200
    batch_size = trial.suggest_int("batch_size", 20, 200, step=60)
    learn_params_torch = {
        "batch_size_train": batch_size,
        "frac_val": 0.3,
        "batch_size_val": batch_size,
        "max_epochs": 5000,
        "shuffle": True,
    }
    model_kwargs = dict(
        optim_cls=optim.Adam,
        optim_params=dict(lr=1e-4),
        # optim_params=dict(lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True)),
    )

    n_layers = 3
    # n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dim = 1000
    # hidden_dim = trial.suggest_inf("n_units", 10, 10000, log=True)
    hidden_dims = n_layers * [hidden_dim]
    # hidden_dims = [trial.suggest_int(f"n_units_l{i}", 10, 10000, log=True) for i in range(n_layers)]
    module = MultiNet.mlp(
        env, hidden_sizes_ch=[], hidden_sizes_tasks=[], hidden_sizes_joint=hidden_dims
    )

    trainer_kwargs = dict(
        logger=TensorBoardLogger(temp_path + "logs/lit/", name=f"{now}_batch-{batch_size}"),
        # logger=TensorBoardLogger(temp_path + "logs/lit/", name=f"{now}_{trial.number}"),
        enable_checkpointing=False,
        # callbacks=EarlyStopping('val_loss', patience=100),
        # callbacks=PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        callbacks=[
            EarlyStopping("val_loss", patience=100),
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        ],
        default_root_dir=temp_path + "logs/lit/",
        accelerator="auto",
        # strategy=DDPStrategy(find_unused_parameters=False),
        strategy=DDPSpawnStrategy(find_unused_parameters=False),
    )

    lit_scheduler = LitScheduler.from_module(
        env,
        module,
        model_kwargs,
        trainer_kwargs=trainer_kwargs,
        learn_params=learn_params_torch,
    )

    # loaded data
    lit_scheduler.train(obs, act, verbose=1)
    # return lit_scheduler.trainer.callback_metrics["val_loss"].item()
    # return lit_scheduler.trainer.callback_metrics["train_loss"].item()
    return lit_scheduler.trainer.callback_metrics["val_acc"].item()

    # # original
    # algorithms = np.array(
    #     [("Lit Policy", lit_scheduler, 10)],
    #     dtype=[("name", "<U32"), ("obj", object), ("n_iter", int)],
    # )

    # n_gen_learn, n_gen = 900, 100
    # # n_gen_learn, n_gen = 80000, 20000
    # # log_path = None
    # log_path = temp_path + "log.md"
    # # img_path = temp_path + f'images/{now}'
    # img_path = None
    # loss_mean, t_run_mean = evaluate_algorithms_gen(
    #     algorithms,
    #     problem_gen,
    #     n_gen,
    #     n_gen_learn,
    #     solve=True,
    #     verbose=1,
    #     plotting=0,
    #     log_path=log_path,
    #     img_path=img_path,
    #     rng=seed,
    # )

    # return loss_mean["Lit Policy"].mean()


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.GridSampler()

    pruner = optuna.pruners.NopPruner()
    # pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    # study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(objective, n_trials=100, timeout=3600 * 10, show_progress_bar=True)

    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    plt.show()

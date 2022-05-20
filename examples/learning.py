from functools import partial

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from torch import nn

from task_scheduling.algorithms import earliest_release, mcts, random_sequencer
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.reinforcement import (
    MultiExtractor,
    StableBaselinesScheduler,
    ValidActorCriticPolicy,
)
from task_scheduling.mdp.supervised.torch import LitScheduler
from task_scheduling.results import evaluate_algorithms_gen, evaluate_algorithms_train

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
seed = 12345

if seed is not None:
    seed_everything(seed)


# Define scheduling problem and algorithms
problem_gen = problem_gens.Random.discrete_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Dataset.load('data/continuous_linear_drop_c1t8', repeat=True)

env_params = {
    "features": None,  # defaults to task parameters
    "sort_func": "t_release",
    "time_shift": True,
    "masking": True,
}

env = Index(problem_gen, **env_params)


learn_params = {
    "batch_size_train": 20,
    "n_gen_val": 1 / 3,
    "batch_size_val": 30,
    "max_epochs": 2000,
    "shuffle": True,
}
trainer_kwargs = {
    "logger": False,
    "enable_checkpointing": False,
    "callbacks": EarlyStopping("val_loss", min_delta=0.0, patience=50),
    "gpus": torch.cuda.device_count(),
}
lit_scheduler = LitScheduler.mlp(
    env,
    hidden_sizes_joint=[400],
    model_kwargs={"optim_params": {"lr": 1e-4}},
    trainer_kwargs=trainer_kwargs,
    learn_params=learn_params,
)


learn_params_sb = {
    "n_gen_val": 1 / 3,
    "max_epochs": 2000,
    "eval_callback_kwargs": dict(
        callback_after_eval=StopTrainingOnNoModelImprovement(1000, min_evals=0, verbose=1),
        n_eval_episodes=100,
        eval_freq=1000,
        verbose=1,
    ),
}
sb_model_kwargs = dict(
    policy=ValidActorCriticPolicy,
    policy_kwargs=dict(
        features_extractor_class=MultiExtractor.mlp,
        features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[]),
        net_arch=[400],
        activation_fn=nn.ReLU,
        normalize_images=False,
        infer_valid_mask=env.infer_valid_mask,
    ),
)
sb_scheduler = StableBaselinesScheduler.make_model(env, "PPO", sb_model_kwargs, learn_params_sb)


algorithms = np.array(
    [
        ("Random", random_sequencer, 10),
        ("ERT", earliest_release, 10),
        (
            "MCTS",
            partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=0.05, th_visit=5),
            10,
        ),
        ("SL Policy", lit_scheduler, 10),
        ("RL Agent", sb_scheduler, 10),
    ],
    dtype=[("name", "<U32"), ("func", object), ("n_iter", int)],
)


# Evaluate results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment

loss_mc, t_run_mc = evaluate_algorithms_train(
    algorithms,
    problem_gen,
    n_gen,
    n_gen_learn,
    n_mc,
    solve=True,
    verbose=1,
    plotting=1,
    rng=seed,
)
loss_mean, t_run_mean = evaluate_algorithms_gen(
    algorithms,
    problem_gen,
    n_gen,
    n_gen_learn,
    solve=True,
    verbose=1,
    plotting=1,
    rng=seed,
)

plt.show()

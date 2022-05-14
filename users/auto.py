from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn

from task_scheduling.algorithms import earliest_release, mcts, random_sequencer
from task_scheduling.base import get_now
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.supervised.torch import LitScheduler, MultiNet, VaryCNN
from task_scheduling.results import evaluate_algorithms_gen, evaluate_algorithms_train

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
plt.style.use("images/style.mplstyle")

now = get_now()

# seed = None
seed = 12345

if seed is not None:
    seed_everything(
        seed
    )  # TODO: doesn't guarantee reproducibility of PL learners if reordered

data_path = Path("data/")


# %% Algorithms
algorithms_base = np.array(
    [
        ("Random", random_sequencer, 10),
        ("ERT", earliest_release, 10),
        *(
            (
                f"MCTS: c={c}, t={t}",
                partial(
                    mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c, th_visit=t
                ),
                10,
            )
            for c, t in product([0], [5, 10])
        ),
    ],
    dtype=[("name", "<U32"), ("func", object), ("n_iter", int)],
)


trainer_kwargs = dict(
    logger=TensorBoardLogger("auto_temp/logs/", name=now),
    enable_checkpointing=False,
    callbacks=EarlyStopping("val_loss", min_delta=1e-3, patience=100),
    default_root_dir="auto_temp/logs/",
    gpus=torch.cuda.device_count(),
)

learn_params = {
    "batch_size_train": 20,
    "n_gen_val": 1 / 3,
    "batch_size_val": 30,
    # 'max_epochs': 1000,
    "max_epochs": 2000,
    "shuffle": True,
}

model_kwargs = dict(
    # optim_params={'lr': 1e-3},
    optim_params={"lr": 1e-4},
)

module_constructors = [
    partial(MultiNet.mlp, hidden_sizes_joint=[400]),
    # partial(MultiNet.cnn, hidden_sizes_tasks=[400], cnn_kwargs=dict(pooling_layers=[nn.AdaptiveMaxPool1d(1)]),
    #         kernel_sizes=2),
    # partial(MultiNet.cnn, hidden_sizes_tasks=[400], cnn_kwargs=dict(pooling_layers=[nn.AdaptiveMaxPool1d(1)]),
    #         kernel_sizes=4),
]

env_params_base = dict(
    normalize=True,
    masking=True,
)

env_params_set = [
    dict(
        sort_func=None,
        time_shift=False,
    ),
    dict(
        sort_func=None,
        time_shift=True,
    ),
    dict(
        sort_func="t_release",
        time_shift=False,
    ),
    dict(
        sort_func="t_release",
        time_shift=True,
    ),
]


# %%
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment

datasets = [
    "continuous_linear_drop_c1t8",
]

# %%
env_params_set = [env_params_base | env_params for env_params in env_params_set]

for dataset in datasets:
    temp_path = f"auto_temp/{dataset}/"
    log_path = temp_path + "log.md"
    img_path = temp_path + f"images/{now}"
    trainer_kwargs["logger"] = TensorBoardLogger(temp_path + "logs/", name=now)

    problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)

    algorithms_data = []
    for (i_env, env_params), (i_net, module_constructor) in product(
        enumerate(env_params_set), enumerate(module_constructors)
    ):
        if seed is not None:
            seed_everything(seed)

        env = Index(problem_gen, **env_params)
        module = module_constructor(env)
        lit_scheduler = LitScheduler.from_module(
            env,
            module,
            model_kwargs,
            trainer_kwargs=trainer_kwargs,
            learn_params=learn_params,
        )

        # name = f"Policy: Env {i_env}, Net {i_net}"
        name = f"Policy: Env {i_env}"

        algorithms_data.append((name, lit_scheduler, 10))

    algorithms_learn = np.array(
        algorithms_data, dtype=[("name", "<U32"), ("func", object), ("n_iter", int)]
    )
    algorithms = np.concatenate((algorithms_base, algorithms_learn))

    loss_mc, t_run_mc = evaluate_algorithms_train(
        algorithms,
        problem_gen,
        n_gen,
        n_gen_learn,
        n_mc,
        solve=True,
        verbose=1,
        plotting=1,
        log_path=log_path,
        img_path=img_path,
        rng=seed,
    )

    # loss_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
    #                                                 verbose=1, plotting=1, log_path=log_path, img_path=img_path,
    #                                                 rng=seed)

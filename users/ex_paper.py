from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from task_scheduling.algorithms import earliest_release, mcts, random_sequencer
from task_scheduling.base import get_now
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.supervised import LitScheduler
from task_scheduling.mdp.util import MultiNet
from task_scheduling.results import evaluate_algorithms_train

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
plt.style.use("images/style.mplstyle")

now = get_now()

# seed = None
seed = 12345

data_path = "data/continuous_linear_drop_c1t8"
save_path = "users/paper_temp/"


# %% Algorithms
algorithms_base = np.array(
    [
        ("Random", random_sequencer, 10),
        ("ERT", earliest_release, 10),
        ("MCTS: T=5", partial(mcts, max_rollouts=10, th_visit=5), 10),
        ("MCTS: T=10", partial(mcts, max_rollouts=10, th_visit=10), 10),
    ],
    dtype=[("name", "<U32"), ("obj", object), ("n_iter", int)],
)

model_kwargs = dict(optim_params=dict(lr=1e-4))
trainer_kwargs = dict(
    logger=TensorBoardLogger(save_path + "logs/", name=now),
    callbacks=EarlyStopping("val_loss", patience=50),
    enable_checkpointing=False,
    log_every_n_steps=10,
    accelerator="auto",
    devices=1,
)
learn_params = {
    "frac_val": 0.3,
    "max_epochs": 5000,
    "dl_kwargs": dict(batch_size=160, shuffle=True),
}

env_params_base = dict(normalize=True, masking=True)
env_params_set = [
    dict(sort_func=None, time_shift=False),
    dict(sort_func=None, time_shift=True),
    dict(sort_func="t_release", time_shift=False),
    dict(sort_func="t_release", time_shift=True),
]


def main():
    if seed is not None:
        seed_everything(seed)

    problem_gen = problem_gens.Dataset.load(data_path, repeat=True)

    policy_data = []
    for i_env, env_params in enumerate(env_params_set):
        env = Index(problem_gen, **env_params_base, **env_params)
        module = MultiNet.mlp(env, hidden_sizes_joint=[400])
        lit_scheduler = LitScheduler.from_module(
            env,
            module,
            model_kwargs,
            trainer_kwargs=deepcopy(trainer_kwargs),
            learn_params=learn_params,
        )
        policy_data.append((f"Policy: Env {i_env}", lit_scheduler, 10))

    algorithms_learn = np.array(
        policy_data, dtype=[("name", "<U32"), ("obj", object), ("n_iter", int)]
    )
    algorithms = np.concatenate((algorithms_base, algorithms_learn))

    evaluate_algorithms_train(
        algorithms,
        problem_gen,
        n_gen=100,
        n_gen_learn=900,
        n_mc=10,
        solve=True,
        verbose=1,
        plotting=1,
        log_path=save_path + "log.md",
        img_path=save_path + f"images/{now}.png",
        rng=seed,
    )


if __name__ == "__main__":
    main()
    plt.show()

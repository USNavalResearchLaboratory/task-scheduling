from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from task_scheduling.base import get_now
from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.learning.supervised.torch import LitScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')

now = get_now()

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)  # TODO: doesn't guarantee reproducibility of PL learners if reordered


# %% Algorithms
algorithms_base = np.array([
    ('Random', random_sequencer, 10),
    ('ERT', earliest_release, 10),
    *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c,
                                      th_visit=t), 10)
      for c, t in product([0], [5, 10])),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


trainer_kwargs = {
    'logger': TensorBoardLogger('auto_temp/logs/', name=now),
    'checkpoint_callback': False,
    'default_root_dir': 'auto_temp/logs/',
    'gpus': min(1, torch.cuda.device_count()),
}

learn_params = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'max_epochs': 1000,
    'shuffle': True,
    'callbacks': EarlyStopping('val_loss', min_delta=0., patience=50),
}

lit_kwargs = {'optim_params': {'lr': 1e-3}}

valid_fwd_set = [False, True]
# valid_fwd_set = [False]

# TODO: generalize for arbitrary models
layer_sizes_set = [
    # [50],
    # [100],
    # [200],
    [400],
    # [50, 50],
    # [100, 100],
    # [200, 200],
    # [400, 400],
]

env_params_set = [
    {
        'sort_func': None,
        'time_shift': False,
        'masking': True,
    },
    # {
    #     'sort_func': None,
    #     'time_shift': True,
    #     'masking': True,
    # },
    # {
    #     'sort_func': 't_release',
    #     'time_shift': False,
    #     'masking': True,
    # },
    {
        'sort_func': 't_release',
        'time_shift': True,
        'masking': True,
    },
]


# %%
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment


data_path = Path('../data/')
datasets = [
    'continuous_relu_drop_c1t8',
    # 'continuous_relu_drop_c2t8',
    # 'discrete_relu_drop_c1t8',
    # 'discrete_relu_drop_c2t8',
]
for dataset in datasets:
    log_path = f'auto_temp/{dataset}/log.md'
    img_path = f"auto_temp/{dataset}/images/{now}.png"

    problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)

    algorithms_data = []
    # for (i_env, env_params), (i_net, layer_sizes) in product(enumerate(env_params_set), enumerate(layer_sizes_set)):
    for (i_env, env_params), (i_net, layer_sizes), valid_fwd in product(enumerate(env_params_set),
                                                                        enumerate(layer_sizes_set),
                                                                        valid_fwd_set):
        if seed is not None:
            seed_everything(seed)

        lit_scheduler = LitScheduler.from_gen_mlp(problem_gen, env_params=env_params, hidden_layer_sizes=layer_sizes,
                                                  lit_kwargs=lit_kwargs, trainer_kwargs=trainer_kwargs,
                                                  learn_params=learn_params, valid_fwd=valid_fwd)

        net_str = str(i_net)
        # net_str = '-'.join(map(str, layer_sizes))

        # algorithms_data.append((f"Policy: Env {i_env}", lit_scheduler, 10))
        # algorithms_data.append((f"Policy: Env {i_env}, MLP {net_str}", lit_scheduler, 10))
        algorithms_data.append((f"Policy: Env {i_env}, Valid={valid_fwd}", lit_scheduler, 10))
        # algorithms_data.append((f"Policy: Env {i_env}, MLP {net_str}, Valid={valid_fwd}", lit_scheduler, 10))

    algorithms_learn = np.array(algorithms_data, dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])
    algorithms = np.concatenate((algorithms_base, algorithms_learn))

    loss_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve=True,
                                                  verbose=1, plotting=1, log_path=log_path, img_path=img_path,
                                                  rng=seed)

    # loss_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
    #                                                 verbose=1, plotting=1, log_path=log_path, img_path=img_path,
    #                                                 rng=seed)

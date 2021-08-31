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

gpus = min(1, torch.cuda.device_count())

now = get_now()

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)


#%% Algorithms

trainer_kwargs = {
    'logger': TensorBoardLogger('logs/learn/', name=now),
    'checkpoint_callback': False,
    'default_root_dir': 'logs/learn',
    'gpus': gpus,
}

learn_params = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'max_epochs': 1000,
    'shuffle': True,
    'callbacks': EarlyStopping('val_loss', min_delta=0., patience=50),
}

lit_mlp_kwargs = {'optim_params': {'lr': 1e-3}}

#
env_params_set = [
    {
        'features': None,  # defaults to task parameters
        'sort_func': None,
        # 'sort_func': 't_release',
        'time_shift': False,
        # 'time_shift': True,
        'masking': False,
        # 'masking': True,
        'action_type': 'valid',
        'seq_encoding': 'one-hot',
    },
    {
        'features': None,  # defaults to task parameters
        # 'sort_func': None,
        'sort_func': 't_release',
        # 'time_shift': False,
        'time_shift': True,
        # 'masking': False,
        'masking': True,
        'action_type': 'valid',
        'seq_encoding': 'one-hot',
    },
]


n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 1  # the number of Monte Carlo iterations performed for scheduler assessment


schedule_path = Path.cwd() / 'data' / 'schedules'

datasets = ['discrete_relu_c1t8']
# datasets = ['discrete_relu_c1t8', 'continuous_relu_c1t8', 'discrete_relu_c2t8', 'continuous_relu_c2t8']
for dataset in datasets:
    problem_gen = problem_gens.Dataset.load(schedule_path / dataset, repeat=True)

    log_path = f'logs/{dataset}.md'
    for env_params in env_params_set:
        img_path = f'images/{dataset}/{now}.png'

        lit_scheduler = LitScheduler.from_env_mlp([30, 30], problem_gen, env_params=env_params,
                                                  lit_mlp_kwargs=lit_mlp_kwargs,
                                                  trainer_kwargs=trainer_kwargs, learn_params=learn_params,
                                                  valid_fwd=True)

        algorithms = np.array([
            ('Random', random_sequencer, 10),
            ('ERT', earliest_release, 10),
            *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c,
                                              visit_threshold=t), 10)
              for c, t in product([0], [5, 10])),
            ('Lit Policy', lit_scheduler, 10),
        ], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])

        l_ex_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
                                                        verbose=1, plotting=1, log_path=log_path, img_path=img_path,
                                                        rng=seed)
        #
        # l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve=True,
        #                                               verbose=1, plotting=1, log_path=log_path, img_path=img_path,
        #                                               rng=seed)

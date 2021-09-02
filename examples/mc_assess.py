from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.learning.supervised.torch import LitScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')

seed = 12345

if seed is not None:
    seed_everything(seed)


#%% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
problem_gen = problem_gens.Dataset.load('../data/schedules/continuous_relu_drop_c1t8', repeat=True)


#%% Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    'sort_func': 't_release',
    'time_shift': True,
    'masking': True,
    'seq_encoding': 'one-hot',
}

learn_params = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'max_epochs': 500,
    'shuffle': True,
    'callbacks': EarlyStopping('val_loss', min_delta=0., patience=50),
}

lit_scheduler = LitScheduler.from_env_mlp([30, 30], problem_gen, env_params=env_params,
                                          lit_mlp_kwargs={'optim_params': {'lr': 1e-3}},
                                          learn_params=learn_params, valid_fwd=True)


algorithms = np.array([
    ('Random', random_sequencer, 10),
    ('ERT', earliest_release, 10),
    ('MCTS', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=.05, visit_threshold=5), 10),
    ('Lit Policy', lit_scheduler, 10),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


#%% Evaluate results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment

l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve=True,
                                              verbose=1, plotting=1, rng=seed)
# l_ex_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
#                                                 verbose=1, plotting=1, rng=seed)

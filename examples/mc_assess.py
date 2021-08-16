from functools import partial
from itertools import product
# from operator import methodcaller

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from torch import nn, optim
from torch.nn import functional
import pytorch_lightning as pl

from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.util import evaluate_algorithms_train
from task_scheduling.learning import environments as envs
from task_scheduling.learning.supervised.torch import LitScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')

seed = 12345


#%% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
problem_gen = problem_gens.Dataset.load('../data/schedules/discrete_relu_c1t8', shuffle=True, repeat=True, rng=seed)


#%% Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    'sort_func': 't_release',
    'time_shift': True,
    'masking': True,
    'action_type': 'valid',
    'seq_encoding': 'one-hot',
}

env = envs.StepTasking(problem_gen, **env_params)


class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(env.observation_space.shape).item(), 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, env.action_space.n),
            nn.LogSoftmax(dim=1),
        )
        self.loss_func = functional.nll_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


learn_params_pl = {'batch_size_train': 20,
                   'n_gen_val': 1/3,
                   'batch_size_val': 30,
                   'weight_func': None,
                   'max_epochs': 400,
                   'shuffle': True,
                   'callbacks': [pl.callbacks.EarlyStopping('val_loss', min_delta=0., patience=100)]
                   }


algorithms = np.array([
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False,
    #                                                                rng=RNGMix.make_rng(seed))), 1),
    ('Random', partial(random_sequencer, rng=seed), 10),
    ('ERT', earliest_release, 10),
    *((f'MCTS: c={c}, t={t}', partial(mcts, runtime=.002, c_explore=c, visit_threshold=t, rng=seed), 10)
      for c, t in product([.035], [15])),
    ('Lit Policy', LitScheduler(env, LitModule(), learn_params_pl), 10),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


#%% Evaluate results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment
l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, n_gen_learn, problem_gen, n_gen=n_gen, n_mc=n_mc, solve=True,
                                              verbose=2, plotting=2, log_path=None)

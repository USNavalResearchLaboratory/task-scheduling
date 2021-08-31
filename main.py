from functools import partial
from itertools import product
from pathlib import Path
from datetime import datetime
# from functools import wraps
# from operator import methodcaller

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from stable_baselines3.common.env_checker import check_env

from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.learning import environments as envs
from task_scheduling.learning.base import Base as BaseLearningScheduler
# from task_scheduling.learning.supervised._tf import keras, Scheduler as tfScheduler
from task_scheduling.learning.supervised.torch import TorchScheduler, LitScheduler
from task_scheduling.learning.reinforcement import StableBaselinesScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

gpus = min(1, torch.cuda.device_count())

now = datetime.now().replace(microsecond=0).isoformat().replace(':', '_')

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)

# TODO: document class attributes, even if identical to init parameters?
# TODO: document instantiation parameters under init or under the class def?
# TODO: rework docstring parameter typing?


#%% Define scheduling problem and algorithms

# gens = {'discrete': problem_gens.Random.discrete_relu_drop,
#         'continuous': problem_gens.Random.continuous_relu_drop}
# for n_tasks in [12, 16]:
#     for name, func in gens.items():
#         save_path = f"data/schedules/{name}_relu_c2t{n_tasks}"
#         gen = func(n_tasks=n_tasks, n_ch=2, rng=seed)
#         list(gen(n_gen=1000, solve=True, verbose=1, save_path=save_path))
# raise Exception

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path.cwd() / 'data'
schedule_path = data_path / 'schedules'

dataset = 'discrete_relu_c1t8'
# dataset = 'discrete_relu_c2t8'
# dataset = 'continuous_relu_c1t8'
# dataset = 'continuous_relu_c2t8'
problem_gen = problem_gens.Dataset.load(schedule_path / dataset, shuffle=True, repeat=True, rng=seed)


# Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    # 'sort_func': None,
    'sort_func': 't_release',
    # 'time_shift': False,
    'time_shift': True,
    # 'masking': False,
    'masking': True,
    'action_type': 'valid',
    'seq_encoding': 'one-hot',
}

env = envs.StepTasking(problem_gen, **env_params)


class TorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(env.observation_space.shape).item(), 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, env.action_space.n),
            nn.Softmax(dim=1),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


model_torch = TorchModule()

loss_func = functional.cross_entropy
# loss_func = functional.nll_loss

# TODO: pass class and kwargs instead?
optimizer = optim.Adam(model_torch.parameters(), lr=1e-3)


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
            nn.Softmax(dim=1),
            # nn.LogSoftmax(dim=1),
        )

        self.loss_func = functional.cross_entropy
        # self.loss_func = functional.nll_loss

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
        # return optim.SGD(self.parameters(), lr=1e-2)


model_pl = LitModule()

pl_trainer_kwargs = {
    'logger': TensorBoardLogger('logs/learn/', name=now),
    'checkpoint_callback': False,
    'default_root_dir': 'logs/learn',
    'gpus': gpus,
    # 'distributed_backend': 'ddp',
    # 'profiler': 'simple',
    # 'progress_bar_refresh_rate': 0,
}

learn_params_torch = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'weight_func': None,  # TODO: weighting based on loss value!?
    # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
    'max_epochs': 50,
    'shuffle': True,
    'callbacks': EarlyStopping('val_loss', min_delta=0., patience=50),
}

valid_fwd = True
# valid_fwd = False


# RL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
#            'model_cls': 'DQN', 'model_params': {'verbose': 1, 'policy': 'MlpPolicy'},
#            'n_episodes': 10000,
#            'save': False, 'save_path': None}
# dqn_agent = StableBaselinesScheduler
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')

# check_env(env)
# # model_cls, model_params = StableBaselinesScheduler.model_defaults['DQN_MLP']
# model_cls, model_params = StableBaselinesScheduler.model_defaults['PPO']
# # model_sb = model_cls(env=env, **model_params)
#
# learn_params_sb = {}


# FIXME: integrate SB3 before making any sweeping environment/learn API changes!!!

# FIXME: no faster on GPU!?!? CHECK batch size effects!
# FIXME: INVESTIGATE huge PyTorch speed-up over Tensorflow!!


algorithms = np.array([
    # ('BB', branch_bound, 1),
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False, rng=seed)), 1),
    # ('BB_p_ERT', partial(branch_bound_priority, heuristic=methodcaller('earliest_release', inplace=False)), 1),
    ('Random', random_sequencer, 10),
    ('ERT', earliest_release, 10),
    *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c, visit_threshold=t), 10)
      for c, t in product([0], [5])),
    # ('TF Policy', tfScheduler(env, model_tf, train_params_tf), 10),
    # ('Torch Policy', TorchScheduler(env, model_torch, loss_func, optimizer, learn_params_torch, valid_fwd), 10),
    ('Lit Policy', LitScheduler(env, model_pl, pl_trainer_kwargs, learn_params_torch, valid_fwd), 10),
    # ('DQN Agent', StableBaselinesScheduler.make_model(env, model_cls, model_params), 5),
    # ('DQN Agent', StableBaselinesScheduler(model_sb, env), 5),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


#%% Evaluate and record results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 1  # the number of Monte Carlo iterations performed for scheduler assessment


# TODO: generate multiple channel results!!
# TODO: show value of valid-action network vs `mask_probability` hack!?


# TODO: generate new, larger datasets
# TODO: try making new features

# TODO: avoid state correlation? Do Env transforms already achieve this?
# TODO: make loss func for full seq targets, penalize in proportion to seq similarity?


log_path = 'logs/temp/PGR_results.md'
# log_path = 'logs/discrete_relu_c1t8.md'

img_path = f'images/temp/{now}.png'


# TODO: summary cleanup

solve = True
# solve = False

l_ex_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve,
                                                verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)

# l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve,
#                                               verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)


# np.savez(data_path / f'results/temp/{now}', l_ex_mc=l_ex_mc, t_run_mc=t_run_mc)


#%% Deprecated

# def make_valid_wrapper(env_):
#     def valid_wrapper(func):
#         @wraps(func)
#         def valid_func(self, *args, **kwargs):
#             p = func(self, *args, **kwargs)
#
#             mask = 1 - env_.make_mask(*args, **kwargs)
#             p_mask = p * mask
#
#             idx_zero = p_mask.sum(dim=1) == 0.
#             p_mask[idx_zero] = mask[idx_zero]  # if no valid actions are non-zero, make them uniform
#
#             p_norm = functional.normalize(p_mask, p=1, dim=1)
#             return p_norm
#
#         return valid_func
#     return valid_wrapper


# tf.random.set_seed(seed)
#
# def _weight_init():
#     return keras.initializers.GlorotUniform(seed)
#
#
# layers = [
#     keras.layers.Flatten(),
#     keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init()),
#     keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init()),
#     # keras.layers.Dropout(0.2),
# ]
#
# # layers = [
# #     keras.layers.Conv1D(30, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
# #     keras.layers.Conv1D(20, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
# #     keras.layers.Conv1D(20, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
# #     # keras.layers.Dense(20, activation='relu', kernel_initializer=_weight_init()),
# #     # keras.layers.Flatten(),
# # ]
#
# # layers = [
# #     keras.layers.Reshape((problem_gen.n_tasks, -1, 1)),
# #     keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer=_weight_init())
# # ]
#
#
# model_tf = keras.Sequential([keras.Input(shape=env.observation_space.shape),
#                              *layers,
#                              keras.layers.Dense(env.action_space.n, activation='softmax',
#                                                 kernel_initializer=_weight_init())])
# model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# train_params_tf = {'batch_size_train': 20,
#                    'n_gen_val': 1/3,
#                    'batch_size_val': 30,
#                    'weight_func': None,
#                    # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
#                    'epochs': 400,
#                    'shuffle': True,
#                    # 'callbacks': [keras.callbacks.EarlyStopping('val_loss', patience=20, min_delta=0.)]
#                    'plot_history': True,
#                    }

# train_args = {'n_batch_train': 30, 'batch_size_train': 20, 'n_batch_val': 10, 'batch_size_val': 30,
#               'weight_func': None,
#               # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
#               'fit_params': {'epochs': 400,
#                              'shuffle': True,
#                              # 'callbacks': [keras.callbacks.EarlyStopping('val_loss', patience=20, min_delta=0.)]
#                              # 'callbacks': [pl.callbacks.EarlyStopping('val_loss', min_delta=0., patience=20)]
#                              },
#               # 'plot_history': True,
#               # 'do_tensorboard': True,
#               }

# n_gen_train = (train_args['n_batch_train'] * train_args['batch_size_train']
#                + train_args['n_batch_val'] * train_args['batch_size_val'])

# sim_type = 'Gen'
# if len(learners) > 0:
#     if isinstance(problem_gen, problem_gens.Dataset) and problem_gen.repeat:
#         problem_gen.repeat = False
#         print('Dataset generator repeat disabled to enforce train/test separation.')
#         problem_gen.shuffle()
#
#     # for learner in learners:
#     #     learner.learn(verbose=2, plot_history=True, **train_args)
#     for func in learners['func']:
#         func.learn(verbose=2, plot_history=True, **train_args)
#
#         # train_path = image_path + '_train'
#         # plt.figure('Training history').savefig(train_path)
#         # with open(log_path, 'a') as fid:
#         #     print(f"![](../{train_path}.png)\n", file=fid)
#
# l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen=100, solve=True, verbose=1, plotting=1,
#                                             log_path=log_path)

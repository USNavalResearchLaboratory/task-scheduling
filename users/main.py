from functools import partial
from itertools import product
from pathlib import Path
# from functools import wraps
# from operator import methodcaller
from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
# from stable_baselines3.common.env_checker import check_env

from task_scheduling.base import get_now
from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.mdp.features import encode_discrete_features
from task_scheduling.mdp.environments import Index, Seq
from task_scheduling.mdp.supervised.torch import TorchScheduler, LitScheduler, ValidNet, VaryCNN
from task_scheduling.mdp.base import RandomAgent
# from task_scheduling.mdp.reinforcement import StableBaselinesScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

now = get_now()

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)

# TODO: document class attributes, even if identical to init parameters?
# TODO: document instantiation parameters under init or under the class def?
# TODO: rework docstring parameter typing?


# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=4, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path('../data/')

dataset = 'continuous_relu_drop_c1t4'
# dataset = 'continuous_relu_drop_c1t8'
# dataset = 'continuous_relu_drop_c2t8'
# dataset = 'discrete_relu_drop_c1t8'
# dataset = 'discrete_relu_drop_c2t8'
problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)


# Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    # 'features': encode_discrete_features(problem_gen),
    # 'sort_func': None,
    'sort_func': 't_release',
    # 'time_shift': False,
    'time_shift': True,
    # 'masking': False,
    'masking': True,
}

env = Index(problem_gen, **env_params)
# env = Seq(problem_gen)


learn_params_torch = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'weight_func': None,
    'max_epochs': 200,
    'shuffle': True,
}


torch_model = VaryCNN(len(env.features))
# torch_model = MultiNet(env, hidden_sizes_joint=[400])

# torch_scheduler = TorchScheduler(env, ValidNet(torch_model), optim_params={'lr': 1e-3}, learn_params=learn_params_torch)
# # torch_scheduler = TorchScheduler.mlp(env, hidden_sizes_joint=[400], optim_params={'lr': 1e-3},
# #                                      learn_params=learn_params_torch)


model_kwargs = {'optim_params': {'lr': 1e-4}}

pl_trainer_kwargs = {
    'logger': TensorBoardLogger('main_temp/logs/', name=now),
    'checkpoint_callback': False,
    'callbacks': EarlyStopping('val_loss', min_delta=1e-3, patience=50),
    'default_root_dir': 'main_temp/logs/',
    'gpus': min(1, torch.cuda.device_count()),
    # 'distributed_backend': 'ddp',
    # 'profiler': 'simple',
    # 'progress_bar_refresh_rate': 0,
}

# lit_scheduler = LitScheduler.from_module(env, ValidNet(torch_model), model_kwargs, trainer_kwargs=pl_trainer_kwargs,
#                                          learn_params=learn_params_torch)
# # lit_scheduler = LitScheduler.mlp(env, hidden_sizes_joint=[400], model_kwargs={'optim_params': {'lr': 1e-3}},
# #                                  trainer_kwargs=pl_trainer_kwargs, learn_params=learn_params_torch)

lit_scheduler = LitScheduler.load('../models/c1t8.mdl', env=env)


random_agent = RandomAgent(env)

# RL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
#            'model_cls': 'DQN', 'model_params': {'verbose': 1, 'policy': 'MlpPolicy'},
#            'n_episodes': 10000,
#            'save': False, 'save_path': None}
# dqn_agent = StableBaselinesScheduler
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')

# env = Index(problem_gen, **env_params)
# check_env(env)
# # model_cls, model_params = StableBaselinesScheduler.model_defaults['DQN_MLP']
# model_cls, model_params = StableBaselinesScheduler.model_defaults['PPO']
# # model_sb = model_cls(env=env, **model_params)
#
# learn_params_sb = {}


algorithms = np.array([
    # ('BB', branch_bound, 1),
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False, rng=seed)), 1),
    # ('BB_p_ERT', partial(branch_bound_priority, heuristic=methodcaller('earliest_release', inplace=False)), 1),
    ('Random', random_sequencer, 10),
    ('ERT', earliest_release, 10),
    # *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c, th_visit=t), 10)
    #   for c, t in product([0], [5, 10])),
    ('Random Agent', random_agent, 10),
    # ('Torch Policy', torch_scheduler, 10),
    ('Lit Policy', lit_scheduler, 10),
    # ('TF Policy', tfScheduler(env, model_tf, train_params_tf), 10),
    # ('DQN Agent', StableBaselinesScheduler.make_model(env, model_cls, model_params), 5),
    # ('DQN Agent', StableBaselinesScheduler(model_sb, env), 5),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


# %% Evaluate and record results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen_learn = 0  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment


# TODO: generate new, larger datasets
# TODO: try making new features
# TODO: sample weighting based on `len(seq_rem)` or execution loss?

# TODO: avoid state correlation? Do Env transforms already achieve this?
# TODO: export masking functionality from `envs` to custom policies!

# TODO: no faster on GPU!?!? CHECK batch size effects!
# TODO: investigate loss curves with/without valid action enforcement


log_path = 'main_temp/log.md'
img_path = f'main_temp/images/{now}.png'

# loss_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve=True,
#                                               verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)

loss_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
                                                verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)


# np.savez(f'main_temp/results/{now}', loss_mc=loss_mc, t_run_mc=t_run_mc)


# %% Deprecated

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
#                    'epochs': 400,
#                    'shuffle': True,
#                    # 'callbacks': [keras.callbacks.EarlyStopping('val_loss', patience=20, min_delta=0.)]
#                    'plot_history': True,
#                    }

# train_args = {'n_batch_train': 30, 'batch_size_train': 20, 'n_batch_val': 10, 'batch_size_val': 30,
#               'weight_func': None,
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
# loss_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen=100, solve=True, verbose=1, plotting=1,
#                                             log_path=log_path)

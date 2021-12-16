from functools import partial
from itertools import product
from pathlib import Path
# from operator import methodcaller
from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from stable_baselines3.common.env_checker import check_env

from task_scheduling.base import get_now
from task_scheduling.algorithms import mcts, random_sequencer, earliest_release
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.mdp.base import RandomAgent
from task_scheduling.mdp.environments import Index
# from task_scheduling.mdp.features import encode_discrete_features
from task_scheduling.mdp.supervised.torch import TorchScheduler, LitScheduler, MultiNet, VaryCNN
from task_scheduling.mdp.supervised.torch.modules import build_mlp
from task_scheduling.mdp.reinforcement import StableBaselinesScheduler, ValidActorCriticPolicy, MultiExtractor


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

now = get_now()

seed = None
# seed = 12345

if seed is not None:
    seed_everything(seed)

# TODO: document class attributes, even if identical to init parameters?
# TODO: document instantiation parameters under init or under the class def?
# TODO: rework docstring parameter typing?

# TODO: generate new, larger datasets
# TODO: try making new features

# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path('../data/')

# dataset = 'continuous_relu_drop_c1t4'
dataset = 'continuous_relu_drop_c1t8'
# dataset = 'continuous_relu_drop_c2t8'
# dataset = 'discrete_relu_drop_c1t8'
# dataset = 'discrete_relu_drop_c2t8'
problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)


# Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    # 'features': encode_discrete_features(problem_gen),
    'normalize': True,
    # 'normalize': False,
    # 'sort_func': None,
    'sort_func': 't_release',
    # 'time_shift': False,
    'time_shift': True,
    # 'masking': False,
    'masking': True,
}

env = Index(problem_gen, **env_params)


learn_params_torch = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'weight_func': None,  # TODO: use reward!?
    # 'weight_func': lambda o, a, r: r,
    # 'weight_func': lambda o, a, r: 1 - o['seq'].sum() / o['seq'].size,
    'max_epochs': 1000,
    'shuffle': True,
}

model_kwargs = {'optim_cls': optim.Adam, 'optim_params': {'lr': 1e-4}}

# torch_model = MultiMLP(env, hidden_sizes_ch=[], hidden_sizes_tasks=[60], hidden_sizes_joint=[400])
# torch_model = MultiNet.mlp(env, hidden_sizes_ch=[], hidden_sizes_tasks=[60], hidden_sizes_joint=[400])
# torch_model = MultiNet.cnn(env, hidden_sizes_ch=[], hidden_sizes_tasks=[400], hidden_sizes_joint=[])
torch_model = VaryCNN(env, kernel_len=2)

# torch_scheduler = TorchScheduler(env, torch_model, **model_kwargs, learn_params=learn_params_torch)
# # torch_scheduler = TorchScheduler.mlp(env, hidden_sizes_joint=[400], **model_kwargs, learn_params=learn_params_torch)


trainer_kwargs = {
    'logger': TensorBoardLogger('main_temp/logs/lit/', name=now),
    'checkpoint_callback': False,
    'callbacks': EarlyStopping('val_loss', min_delta=1e-3, patience=100),
    'default_root_dir': 'main_temp/logs/pl/',
    'gpus': min(1, torch.cuda.device_count()),
    # 'distributed_backend': 'ddp',
    # 'profiler': 'simple',
    # 'progress_bar_refresh_rate': 0,
}

lit_scheduler = LitScheduler.from_module(env, torch_model, model_kwargs, trainer_kwargs=trainer_kwargs,
                                         learn_params=learn_params_torch)
# lit_scheduler = LitScheduler.mlp(env, hidden_sizes_joint=[400], model_kwargs=model_kwargs,
#                                  trainer_kwargs=trainer_kwargs, learn_params=learn_params_torch)

# lit_scheduler = LitScheduler.load('../models/c1t8.pth', env=env)
# lit_scheduler = LitScheduler.load('../models/c1t8.pth', trainer_kwargs={'logger': False})  # FIXME


random_agent = RandomAgent(env)

# TODO: imitation learning

# TODO: stopping callbacks
# TODO: more tensorboard, add path to my log

# check_env(env)

learn_params_sb = {
    'max_epochs': 1,  # TODO: check
}

model_params = {
    'policy': ValidActorCriticPolicy,
    'policy_kwargs': dict(
        features_extractor_class=MultiExtractor.mlp,
        features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[60]),
        net_arch=[400],
        activation_fn=nn.ReLU,
        normalize_images=False,
        infer_valid_mask=env.infer_valid_mask,
    ),
    'n_steps': 2048,  # TODO: investigate problem reuse
    'tensorboard_log': 'main_temp/logs/sb/',
    'verbose': 1,
}
sb_scheduler = StableBaselinesScheduler.make_model(env, 'PPO', model_params, learn_params_sb)
# sb_scheduler = StableBaselinesScheduler.make_model(env, 'A2C', model_params, learn_params_sb)


#
algorithms = np.array([
    # ('BB', branch_bound, 1),
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False)), 1),
    # ('BB_p_ERT', partial(branch_bound_priority, heuristic=methodcaller('earliest_release', inplace=False)), 1),
    ('Random', random_sequencer, 10),
    ('ERT', earliest_release, 10),
    # *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=np.inf, max_rollouts=10, c_explore=c, th_visit=t), 10)
    #   for c, t in product([0], [5, 10])),
    ('Random Agent', random_agent, 10),
    # ('Torch Policy', torch_scheduler, 10),
    # ('Lit Policy', lit_scheduler, 10),
    # ('TF Policy', tfScheduler(env, model_tf, train_params_tf), 10),
    ('SB Agent', sb_scheduler, 5),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


# %% Evaluate and record results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment


# TODO: no faster on GPU!?!? CHECK batch size effects!
# TODO: investigate loss curves with/without valid action enforcement
# TODO: avoid state correlation? Do Env transforms already achieve this?


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

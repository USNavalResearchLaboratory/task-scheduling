from functools import partial
from itertools import product
from time import strftime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn, optim

from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_train
from task_scheduling.util.generic import RandomGeneratorMixin as RNGMix
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.algorithms import free
from task_scheduling.learning.supervised.tf import Scheduler as tfScheduler
from task_scheduling.learning.supervised.torch import Scheduler as torchScheduler
from task_scheduling.learning import environments as envs


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

time_str = strftime('%Y-%m-%d_%H-%M-%S')

# seed = None
seed = 12345

# tf.random.set_seed(seed)


# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=4, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path.cwd() / 'data'
schedule_path = data_path / 'schedules'

# list(problem_gen(100, save_path=schedule_path/'temp'/time_str))

dataset = 'discrete_relu_c1t8'
# dataset = 'continuous_relu_c1t8'
# dataset = 'search_track_c1t8_release_0'

problem_gen = problem_gens.Dataset.load(schedule_path / dataset, shuffle=False, repeat=True, rng=seed)


# Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    # 'sort_func': None,
    'sort_func': 't_release',
    # 'time_shift': False,
    'time_shift': True,
    # 'masking': False,
    'masking': True,
    'action_type': 'any',
    # 'seq_encoding': None,
    'seq_encoding': 'one-hot',
}

env = envs.StepTasking(problem_gen, **env_params)


def _weight_init():
    return keras.initializers.GlorotUniform(seed)


layers = [keras.layers.Flatten(),
          keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init()),
          keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init()),
          # keras.layers.Dropout(0.2),
          ]

# layers = [keras.layers.Conv1D(30, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
#           keras.layers.Conv1D(20, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
#           keras.layers.Conv1D(20, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
#           # keras.layers.Dense(20, activation='relu', kernel_initializer=_weight_init()),
#           keras.layers.Flatten(),
#           ]

# layers = [keras.layers.Reshape((problem_gen.n_tasks, -1, 1)),
#           keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer=_weight_init())]


model = keras.Sequential([keras.Input(shape=env.observation_space.shape),
                          *layers,
                          keras.layers.Dense(env.action_space.n, activation='softmax',
                                             kernel_initializer=_weight_init())
                          ])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


train_args = {'n_batch_train': 30, 'batch_size_train': 20, 'n_batch_val': 10, 'batch_size_val': 30,
              'weight_func': None,
              # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
              'fit_params': {'epochs': 100,
                             'callbacks': [keras.callbacks.EarlyStopping('val_loss', patience=200, min_delta=0.)]
                             },
              }


# RL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
#            'model_cls': 'DQN', 'model_params': {'verbose': 1, 'policy': 'MlpPolicy'},
#            'n_episodes': 10000,
#            'save': False, 'save_path': None}
# dqn_agent = learning.RL_policy.ReinforcementLearningScheduler.train_from_gen(**RL_args)
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')


model_torch = nn.Sequential(
    nn.Flatten(),
    nn.Linear(np.prod(env.observation_space.shape).item(), 30),
    nn.ReLU(),
    nn.Linear(30, 30),
    nn.ReLU(),
    nn.Linear(30, env.action_space.n),
    # nn.Softmax(dim=1),
)

loss_func = nn.CrossEntropyLoss()
opt = optim.SGD(model_torch.parameters(), lr=1e-2)

# FIXME: no faster on GPU!?!?
# FIXME: INVESTIGATE huge PyTorch speed-up over Tensorflow!!

# TODO: SL base class
# TODO: new MCTS parameter search for shorter runtime

algorithms = np.array([
    # ('BB', partial(free.branch_bound, rng=RNGMix.make_rng(seed)), 1),
    # ('BB_p', partial(free.branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False, rng=RNGMix.make_rng(seed))), 1),
    # ('BB_p_ERT', partial(free.branch_bound_priority, heuristic=methodcaller('earliest_release', inplace=False)), 1),
    # ('B&B sort', sort_wrapper(partial(free.branch_bound, verbose=False), 't_release'), 1),
    # ('Ensemble', ensemble_scheduler(free.random_sequencer, free.earliest_release), 5),
    ('Random', partial(free.random_sequencer, rng=RNGMix.make_rng(seed)), 10),
    ('ERT', free.earliest_release, 10),
    *((f'MCTS, c={c}, t={t}', partial(free.mcts, runtime=.002, c_explore=c, visit_threshold=t,
                                      rng=RNGMix.make_rng(seed)), 10) for c, t in product([.035], [15])),
    # *((f'MCTS_v1, c={c}', partial(free.mcts_v1, runtime=.02, c_explore=c, rng=RNGMix.make_rng(seed)), 10) for c in [15]),
    # *((f'MCTS, c={c}, t={t}', partial(free.mcts, n_mc=50, c_explore=c, visit_threshold=t,
    #                                   rng=RNGMix.make_rng(seed)), 10) for c, t in product([0.05], [15])),
    # *((f'MCTS_v1, c={c}', partial(free.mcts_v1, n_mc=50, c_explore=c, rng=RNGMix.make_rng(seed)), 10) for c in [10]),
    # ('NN Policy', tfScheduler(env, model), 10),
    ('NN Policy', torchScheduler(env, model_torch, loss_func, opt), 10),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


# %% Evaluate and record results

# TODO: generate new, larger datasets
# TODO: try making new features
# TODO: make problem a shared node class attribute? Setting them seems hackish...
# TODO: value networks


log_path = 'docs/temp/PGR_results.md'
# log_path = 'docs/discrete_relu_c1t8.md'

image_path = f'images/temp/{time_str}'


with open(log_path, 'a') as fid:
    print(f"\n# {time_str}\n", file=fid)
    # print(f"Problem gen: ", end='', file=fid)
    # problem_gen.summary(fid)

    if 'NN Policy' in algorithms['name']:
        idx_nn = algorithms['name'].tolist().index('NN Policy')
        algorithms['func'][idx_nn].summary(fid)
        n_gen_train = (train_args['n_batch_train'] * train_args['batch_size_train']
                       + train_args['n_batch_val'] * train_args['batch_size_val'])
        print(f"Training problems = {n_gen_train}\n", file=fid)

    print('Results\n---', file=fid)


sim_type = 'Gen'
if 'NN Policy' in algorithms['name']:
    if isinstance(problem_gen, problem_gens.Dataset) and problem_gen.repeat:
        problem_gen.repeat = False
        print('Dataset generator repeat disabled to enforce train/test separation.')

    idx_nn = algorithms['name'].tolist().index('NN Policy')
    algorithms['func'][idx_nn].learn(verbose=2, plot_history=True, **train_args)

    train_path = image_path + '_train'
    plt.figure('Training history').savefig(train_path)
    with open(log_path, 'a') as fid:
        print(f"![](../{train_path}.png)\n", file=fid)
l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen=100, solve=True, verbose=1, plotting=1,
                                            log_path=log_path)


# sim_type = 'Train'
# l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, train_args, problem_gen, n_gen=100, n_mc=10, solve=True,
#                                               verbose=3, plotting=1, log_path=log_path)
# np.savez(data_path / f'results/temp/{time_str}', l_ex_mc=l_ex_mc, t_run_mc=t_run_mc)


# plt.figure(f'{sim_type}').savefig(image_path)
plt.figure(f'{sim_type} (Relative)').savefig(image_path)
with open(log_path, 'a') as fid:
    print(f"![](../{image_path}.png)\n", file=fid)
    # str_ = image_path.resolve().as_posix().replace('.png', '')s


# %% Limited Runtime

# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     ('Random', runtime_wrapper(algs.free.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs.free.earliest_release), 1),
#     ('MCTS', partial(algs.limit.mcts, verbose=False), 5),
#     ('DNN Policy', runtime_wrapper(policy_model), 5),
#     # ('DQN Agent', dqn_agent, 5),
# ], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
#
# runtimes = np.logspace(-2, -1, 20, endpoint=False)
# evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=40, solve=True, verbose=2, plotting=1,
#                             save=False, file=None)

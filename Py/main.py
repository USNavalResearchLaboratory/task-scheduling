from functools import partial
from time import strftime
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime, evaluate_algorithms_train
from task_scheduling.util.generic import RandomGeneratorMixin as RNGMix
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.algorithms import free
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler
from task_scheduling.learning import environments as envs
from task_scheduling.learning.features import param_features, encode_discrete_features
from tests import seq_num_encoding


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)  # TODO: compatibility issue workaround

time_str = strftime('%Y-%m-%d_%H-%M-%S')

# seed = None
seed = 12345

# tf.random.set_seed(seed)


# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=4, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)
# problem_gen = problem_gens.Dataset.load('data/continuous_relu_c1t8', shuffle=True, repeat=False, rng=seed)

# problem_gen = problem_gens.Dataset.load('data/discrete_relu_c1t8', shuffle=True, repeat=False, rng=seed)
problem_gen = problem_gens.Dataset.load('data/discrete_relu_c1t8', shuffle=False, repeat=True, rng=seed)

# problem_gen = problem_gens.Dataset.load('data/search_track_c1t8_release_0', shuffle=True, repeat=False, rng=seed)


# TODO: deprecate? Random gen RNG sharing defeats my original reproducibility purpose...
# if isinstance(problem_gen, problem_gens.Dataset):
#     # Pop evaluation problems for new dataset generator
#     problem_gen, problem_gen_train = problem_gen.pop_dataset(n_gen, shuffle=True, repeat=False, rng=seed), problem_gen
# else:
#     problem_gen_train = deepcopy(problem_gen)  # copy random generator
#     problem_gen_train.rng = problem_gen.rng  # share RNG, avoid train/test overlap


# Algorithms

features = None
# features = param_features(problem_gen, time_shift)
# features = encode_discrete_features(problem_gen)

# sort_func = None
sort_func = 't_release'
# def sort_func(task):
#     return task.t_release

# time_shift = False
time_shift = True

# masking = False
masking = True

# seq_encoding = None
seq_encoding = 'one-hot'

# env_cls = envs.SeqTasking
env_cls = envs.StepTasking

env_params = {'features': features,
              'sort_func': sort_func,
              'time_shift': time_shift,
              'masking': masking,
              # 'action_type': 'int',
              'action_type': 'any',
              'seq_encoding': seq_encoding,
              }

env = env_cls(problem_gen, **env_params)


def _weight_init():
    return keras.initializers.GlorotUniform(seed)


layers = [keras.layers.Flatten(),
          keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init()),
          # keras.layers.Dropout(0.2),
          ]

# layers = [keras.layers.Conv1D(50, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
#           keras.layers.Conv1D(20, kernel_size=2, activation='relu', kernel_initializer=_weight_init()),
#           # keras.layers.Dense(20, activation='relu', kernel_initializer=_weight_init()),
#           ]

# layers = [keras.layers.Reshape((problem_gen.n_tasks, -1, 1)),
#           keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer=_weight_init())]

model = keras.Sequential([keras.Input(shape=env.observation_space.shape),
                          *layers,
                          keras.layers.Dense(env.action_space.n, activation='softmax',
                                             kernel_initializer=_weight_init())
                          ])
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


train_args = {'n_batch_train': 30, 'n_batch_val': 15, 'batch_size': 20,
              'weight_func': None,
              # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
              'fit_params': {'epochs': 500},
              'do_tensorboard': False,
              'plot_history': True,
              }

policy_model = SupervisedLearningScheduler(model, env)
policy_model.learn(**train_args)


# RL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
#            'model_cls': 'DQN', 'model_params': {'verbose': 1, 'policy': 'MlpPolicy'},
#            'n_episodes': 10000,
#            'save': False, 'save_path': None}
# dqn_agent = learning.RL_policy.ReinforcementLearningScheduler.train_from_gen(**RL_args)
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    ('Random', partial(free.random_sequencer, rng=RNGMix.make_rng(seed)), 10),
    ('ERT', free.earliest_release, 1),
    ('MCTS', partial(free.mcts, n_mc=50, rng=RNGMix.make_rng(seed)), 10),
    ('NN', policy_model, 1),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])


# %% Evaluate and record results

if not isinstance(problem_gen, problem_gens.Dataset):
    problem_gens.Base.temp_path = 'data/temp/'  # set a temp path for saving new data

# log_path = None
# log_path = 'docs/temp/PGR_results.md'
log_path = 'docs/discrete_relu_c1t8.md'

image_path = f'images/temp/{time_str}'

with open(log_path, 'a') as fid:
    print(f"\n# {time_str}\n", file=fid)
    # print(f"Problem gen: ", end='', file=fid)
    # problem_gen.summary(fid)

    if 'NN' in algorithms['name']:
        policy_model.summary(fid)
        train_path = image_path + '_train'
        plt.figure('Training history').savefig(train_path)
        print(f"\n![](../{train_path}.png)\n", file=fid)

    print('Results\n---\n', file=fid)

# l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen=10, solve=True, verbose=1, plotting=1,
#                                             log_path=log_path)
l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, train_args, problem_gen, n_gen=10, n_mc=2, solve=True,
                                              verbose=2, plotting=1, log_path=log_path)

# plt.figure('Gen (Relative)').savefig(image_path)
# plt.figure('Gen (Relative, opt excluded)').savefig(image_path)
plt.figure('Train (Relative, opt excluded)').savefig(image_path)
with open(log_path, 'a') as fid:
    # str_ = image_path.resolve().as_posix().replace('.png', '')
    print(f"![](../{image_path}.png)\n", file=fid)


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

from functools import partial
from time import strftime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from task_scheduling.util.results import evaluate_algorithms_train
from task_scheduling.util.generic import RandomGeneratorMixin as RNGMix, reset_weights
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.algorithms import free
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler
from task_scheduling.learning import environments as envs
from task_scheduling.util.plot import plot_task_losses, plot_schedule, scatter_loss_runtime, plot_loss_runtime
from task_scheduling.learning.features import param_features, encode_discrete_features
from tests import seq_num_encoding


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')
# plt.rc('axes', grid=True)

for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)      # TODO: compatibility issue workaround

time_str = strftime('%Y-%m-%d_%H-%M-%S')


# seed = None
seed = 12345

# rng = np.random.default_rng(seed)

# tf.random.set_seed(seed)


#%% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=4, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=rng)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=rng)
# problem_gen = problem_gens.Dataset.load('data/continuous_relu_c1t8', shuffle=True, repeat=False, rng=rng)

problem_gen = problem_gens.Dataset.load('data/discrete_relu_c1t8', shuffle=False, repeat=True, rng=seed)

# problem_gen = problem_gens.Dataset.load('data/search_track_c1t8_release_0', shuffle=True, repeat=False, rng=rng)


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

model = keras.Sequential([keras.Input(shape=env.observation_space.shape),
                          *layers,
                          keras.layers.Dense(env.action_space.n, activation='softmax',
                                             kernel_initializer=_weight_init())
                          ])
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

policy_model = SupervisedLearningScheduler(model, env)


train_args = {'n_batch_train': 30, 'n_batch_val': 15, 'batch_size': 2,
              'weight_func': None,
              # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
              'fit_params': {'epochs': 50},
              'plot_history': True,
              }


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    ('Random', partial(free.random_sequencer, rng=RNGMix.make_rng(seed)), 10),
    ('ERT', free.earliest_release, 1),
    ('MCTS', partial(free.mcts, n_mc=50, rng=RNGMix.make_rng(seed)), 10),
    ('NN', policy_model, 1),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])


#%% Evaluate and record results

evaluate_algorithms_train(algorithms, train_args, problem_gen, n_gen=10, n_mc=2, solve=True, verbose=2, plotting=1,
                          log_path=None)


from functools import partial
from time import strftime
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling import algorithms as algs
from task_scheduling import learning
from task_scheduling.learning import environments as envs
from task_scheduling.learning.features import param_features, encode_discrete_features
from tests import seq_num_encoding

# TODO: reconsider init imports - dont want TF overhead if unneeded?

plt.style.use('seaborn')
# plt.rc('axes', grid=True)

np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format

for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)      # TODO: compatibility issue workaround

time_str = strftime('%Y-%m-%d_%H-%M-%S')


# seed = None
seed = 12345

rng = np.random.default_rng(seed)

# tf.random.set_seed(seed)


#%% Define scheduling problem and algorithms


# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=4, n_ch=1, rng=rng)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=rng)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=rng)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=rng)
# problem_gen = problem_gens.Dataset.load('data/continuous_relu_c1t8', shuffle=True, repeat=False, rng=rng)
# problem_gen = problem_gens.Dataset.load('data/discrete_relu_c1t8', shuffle=True, repeat=False, rng=rng)
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


env = env_cls.from_problem_gen(problem_gen, env_params)


_weight_init = 'glorot_uniform'
# _weight_init = keras.initializers.GlorotUniform(seed)

layers = [keras.layers.Flatten(),
          keras.layers.Dense(30, activation='relu', kernel_initializer=_weight_init),
          # keras.layers.Dropout(0.2),
          ]

model = keras.Sequential([keras.Input(shape=env.observation_space.shape),
                          *layers,
                          keras.layers.Dense(env.action_space.n, activation='softmax', kernel_initializer=_weight_init)
                          ])
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

policy_model = learning.SL_policy.SupervisedLearningScheduler(model, env)


SL_args = {'n_batch_train': 30, 'n_batch_val': 15, 'batch_size': 2,
           'weight_func': None,
           # 'weight_func': lambda env_: 1 - len(env_.node.seq) / env_.n_tasks,
           'fit_params': {'epochs': 100},
           'plot_history': True,
           }


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    ('Random', partial(algs.free.random_sequencer, rng=rng), 10),
    ('ERT', algs.free.earliest_release, 1),
    ('MCTS', partial(algs.free.mcts, n_mc=50, rng=rng), 10),
    ('NN', policy_model, 1),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])


#%% Evaluate and record results

n_mc = 2
for i_mc in range(n_mc):

    # if isinstance(problem_gen, problem_gens.Dataset):
    #     # Pop evaluation problems for new dataset generator
    #     problem_gen, problem_gen_train = problem_gen.pop_dataset(n_gen, shuffle=True, repeat=False, rng=seed), problem_gen
    # else:
    #     problem_gen_train = deepcopy(problem_gen)  # copy random generator
    #     problem_gen_train.rng = problem_gen.rng  # share RNG, avoid train/test overlap

    # Train supervised learner
    _idx = algorithms['name'].tolist().index('NN')
    algorithms['func'][_idx].learn(**SL_args)

    l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=10, solve=True, verbose=1, plotting=1,
                                                data_path=None, log_path=None)

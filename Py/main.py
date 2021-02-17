from functools import partial
from time import strftime
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling import algorithms as algs
from task_scheduling import learning
from task_scheduling.learning import environments as envs
from task_scheduling.learning.features import param_features, encode_discrete_features
from tests import seq_num_encoding

plt.style.use('seaborn')
# plt.rc('axes', grid=True)

time_str = strftime('%Y-%m-%d_%H-%M-%S')

np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format


#%% Define scheduling problem and algorithms

# seed = None
seed = 100

n_gen = 100

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=4, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)
# problem_gen = problem_gens.Dataset.load('data/continuous_relu_c1t8', shuffle=True, repeat=False, rng=seed)
problem_gen = problem_gens.Dataset.load('data/discrete_relu_c1t8', shuffle=True, repeat=False, rng=seed)
# problem_gen = problem_gens.Dataset.load('data/search_track_c1t8_release_0', shuffle=True, repeat=False, rng=seed)


if isinstance(problem_gen, problem_gens.Dataset):
    # Pop evaluation problems for new dataset generator
    _temp = problem_gen.pop_dataset(n_gen, shuffle=True, repeat=False, rng=seed)
    problem_gen_train = problem_gen
    problem_gen = _temp
else:
    # Copy, re-seed random generator
    problem_gen_train = deepcopy(problem_gen)
    problem_gen_train.rng = seed


# Algorithms

features = None
# features = param_features(problem_gen, time_shift)
# features = encode_discrete_features(problem_gen)

# sort_func = None
sort_func = 't_release'
# sort_func = 'duration'
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

# layers = None
layers = [keras.layers.Flatten(),
          keras.layers.Dense(30, activation='relu'),
          # keras.layers.Dense(30, activation='relu'),
          # keras.layers.Dropout(0.2),
          ]

# layers = [keras.layers.Conv1D(30, kernel_size=2, activation='relu'),
#           ]

# layers = [keras.layers.Reshape((problem_gen.n_tasks, -1, 1)),
#           keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu')]


weight_func_ = None
# def weight_func_(env):
#     return 1 - len(env.node.seq) / env.n_tasks


SL_args = {'problem_gen': problem_gen_train, 'env_cls': env_cls, 'env_params': env_params,
           'layers': layers,
           'n_batch_train': 30, 'n_batch_val': 15, 'batch_size': 20,
           'weight_func': weight_func_,
           'fit_params': {'epochs': 500},
           'plot_history': True,
           'save': False, 'save_path': None}
policy_model = learning.SL_policy.SupervisedLearningScheduler.train_from_gen(**SL_args)
# policy_model = SL_Scheduler.load('temp/2020-10-28_14-56-42')


# RL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
#            'model_cls': 'DQN', 'model_params': {'verbose': 1, 'policy': 'MlpPolicy'},
#            'n_episodes': 10000,
#            'save': False, 'save_path': None}
# dqn_agent = learning.RL_policy.ReinforcementLearningScheduler.train_from_gen(**RL_args)
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    # ('Random', partial(algs.free.random_sequencer, rng=seed), 20),
    # ('ERT', algs.free.earliest_release, 1),
    # ('MCTS', partial(algs.free.mcts, n_mc=50, rng=seed), 5),
    ('DNN', policy_model, 5),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])


#%% Evaluate and record results

if not isinstance(problem_gen, problem_gens.Dataset):
    problem_gens.Base.temp_path = 'data/temp/'  # set a temp path for saving new data


# log_path = None
# log_path = 'docs/temp/PGR_results.md'
log_path = 'docs/discrete_relu_c1t8.md'

image_path = f'images/temp/{time_str}'


with open(log_path, 'a') as fid:
    print(f"# {time_str}\n", file=fid)
    # print(f"Problem gen: ", end='', file=fid)
    # problem_gen.summary(fid)
    if 'DNN' in algorithms['name']:
        policy_model.summary(fid)
        train_path = image_path + '_train'
        plt.figure('Training history').savefig(train_path)
        print(f"\n![](../{train_path}.png)\n", file=fid)
    print('Results\n---', file=fid)

l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen, solve=True, verbose=1, plotting=1,
                                            data_path=None, log_path=log_path)

# plt.figure('Results (Normalized)').savefig(image_path)
plt.figure('Results (Normalized, BB excluded)').savefig(image_path)
with open(log_path, 'a') as fid:
    # str_ = image_path.resolve().as_posix().replace('.png', '')
    print(f"\n![](../{image_path}.png)\n", file=fid)


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

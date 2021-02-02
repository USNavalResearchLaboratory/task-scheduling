from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from task_scheduling.util.generic import runtime_wrapper
from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling import algorithms as algs
from task_scheduling import learning
from task_scheduling.learning import environments as envs
from task_scheduling.learning.features import param_features, encode_discrete_features

# n_tasks = 8
# n_ch = 1
#
# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.))
# filename = f"continuous_relu_c{n_ch}t{n_tasks}"
# list(problem_gen(n_gen=1000, solve=True, verbose=True, save=True, file=filename, rng=None))
#
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.))
# filename = f"discrete_relu_c{n_ch}t{n_tasks}"
# list(problem_gen(n_gen=1000, solve=True, verbose=True, save=True, file=filename, rng=None))
#
# t_r_maxes = [0, .018, .036]
# for t_r_max in t_r_maxes:
#     problem_gen = problem_gens.Random.search_track(n_tasks, n_ch, t_release_lim=(0., t_r_max), ch_avail_lim=(0., 0.))
#     filename = f"search_track_c{n_ch}t{n_tasks}_release_{t_r_max*1e3:.0f}"
#     list(problem_gen(n_gen=1000, solve=True, verbose=True, save=True, file=filename, rng=None))


#%%

# NOTE: ensure train/test separation for loaded data, use iter_mode='once'
# NOTE: to train multiple schedulers on same loaded data, use problem_gen.restart(shuffle=False)

# FIXME: use seeding for comparison results!!

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), ch_avail_lim=(0., 0.))
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=16, n_ch=1, rng=None)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2))
problem_gen = problem_gens.Dataset.load('continuous_relu_c1t12', shuffle=True, repeat=False, rng=None)
# problem_gen = problem_gens.Dataset.load('discrete_relu_c1t12', shuffle=True, repeat=False, rng=None)
# problem_gen = problem_gens.Dataset.load('search_track_c1t8_release_36', shuffle=True, repeat=False, rng=None)


# Algorithms

features = None
# features = param_features(problem_gen, time_shift)
# features = encode_discrete_features(problem_gen)

sort_func = None
# sort_func = 't_release'
# def sort_func(task):
#     return task.t_release

# time_shift = False
time_shift = True

# masking = False
masking = True

# env_cls = envs.SeqTasking
env_cls = envs.StepTasking

env_params = {'features': features,
              'sort_func': sort_func,
              'time_shift': time_shift,
              'masking': masking,
              # 'action_type': 'int',
              'action_type': 'any',
              'seq_encoding': 'one-hot',
              }

# layers = None
layers = [keras.layers.Flatten(),
          keras.layers.Dense(20, activation='relu'),
          # keras.layers.Dense(30, activation='relu'),
          # keras.layers.Dropout(0.2),
          ]

# layers = [keras.layers.Conv1D(30, kernel_size=2, activation='relu'),
#           ]

# layers = [keras.layers.Reshape((problem_gen.n_tasks, -1, 1)),
#           keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu')]

weight_func_ = None
# def weight_func_(env):
#     return (env.n_tasks - len(env.node.seq)) / env.n_tasks

SL_args = {'problem_gen': problem_gen, 'env_cls': env_cls, 'env_params': env_params,
           'layers': layers,
           'n_batch_train': 35, 'n_batch_val': 10, 'batch_size': 20,
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
    ('Random', algs.free.random_sequencer, 20),
    ('ERT', algs.free.earliest_release, 1),
    ('MCTS', partial(algs.free.mcts, n_mc=60, verbose=False), 5),
    ('DNN Policy', policy_model, 5),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=40, solve=True, verbose=1, plotting=1,
                                            save=(not isinstance(problem_gen, problem_gens.Dataset)), file=None)


# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     ('Random', runtime_wrapper(algs.free.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs.free.earliest_release), 1),
#     ('MCTS', partial(algs.limit.mcts, verbose=False), 5),
#     ('DNN Policy', runtime_wrapper(policy_model), 5),
#     # ('DQN Agent', dqn_agent, 5),
# ], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])
#
# runtimes = np.logspace(-2, -1, 20, endpoint=False)
# evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=40, solve=True, verbose=2, plotting=1,
#                             save=False, file=None)

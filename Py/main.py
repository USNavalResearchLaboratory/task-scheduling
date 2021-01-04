from functools import partial
import cProfile
import pstats

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from task_scheduling.util.generic import runtime_wrapper, make_attr_feature
from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling import tree_search
from task_scheduling.algorithms import base as algs_base
from task_scheduling.algorithms import runtime as algs_timed
from task_scheduling import learning
from task_scheduling.learning import environments as envs
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler as SL_Scheduler


#%%

# NOTE: ensure train/test separation for loaded data, use iter_mode='once'
# NOTE: to train multiple schedulers on same loaded data, use problem_gen.restart(shuffle=False)

# problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.DeterministicTasks.continuous_relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.PermutedTasks.continuous_relu_drop(n_tasks=16, n_ch=1, rng=None)
problem_gen = problem_gens.Dataset.load('relu_c1t8_1000', shuffle=True, repeat=False, rng=None)
# problem_gen = problem_gens.DatasetOld.load('discrete_relu_c1t8_1000', iter_mode='once', shuffle_mode='once', rng=None)
# problem_gen = problem_gens.Dataset.load('search_track_c1t8_1000', shuffle=True, repeat=False, rng=None)
# problem_gen = problem_gens.Random.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.01))
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2))

# Algorithms
features = np.array([('duration', make_attr_feature('duration'), problem_gen.task_gen.param_lims['duration']),
                     ('t_release', make_attr_feature('t_release'),
                      (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     ('t_r_rel', lambda tasks, ch_avail: [task.t_release - np.min(ch_avail) for task in tasks],
                      (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     ('slope', make_attr_feature('slope'), problem_gen.task_gen.param_lims['slope']),
                     ('t_drop', make_attr_feature('t_drop'), (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                     ('l_drop', make_attr_feature('l_drop'), (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                     # ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                     # ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])


# sort_func = None
sort_func = 't_release'
# def sort_func(task):
#     return task.t_release

weight_func_ = None
# def weight_func_(env):
#     return (env.n_tasks - len(env.node.seq)) / env.n_tasks

# env_cls = envs.SeqTasking
env_cls = envs.StepTasking

env_params = {'features': features,
              'sort_func': sort_func,
              'time_shift': True,
              'masking': True,
              # 'action_type': 'int',
              'action_type': 'any',
              'seq_encoding': 'one-hot',
              }

# layers = None
layers = [keras.layers.Dense(30, activation='relu'),
          # keras.layers.Dense(10, activation='relu'),
          # keras.layers.Dense(30, activation='relu'),
          # keras.layers.Dropout(0.2),
          # keras.layers.Dense(100, activation='relu'),
          ]

policy_model = SL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, layers=layers, compile_params=None,
                                           n_batch_train=35, n_batch_val=10, batch_size=20, weight_func=weight_func_,
                                           fit_params={'epochs': 400}, do_tensorboard=False, plot_history=True,
                                           save=False, save_path=None)
# policy_model = SL_Scheduler.load('temp/2020-10-28_14-56-42')


# random_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, model_cls='Random', n_episodes=1)
# RL_args = {'problem_gen': problem_gen,
#            'env_cls': env_cls,
#            'env_params': env_params,
#            'model_cls': 'DQN',
#            'model_params': {'verbose': 1},
#            'n_episodes': 100,
#            'save': False,
#            'save_path': None}
# dqn_agent = learning.RL_policy.ReinforcementLearningScheduler.train_from_gen(**RL_args)
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    ('Random', algs_base.random_sequencer, 20),
    ('ERT', algs_base.earliest_release, 1),
    ('MCTS', partial(algs_base.mcts, n_mc=100, verbose=False), 5),
    ('DNN Policy', policy_model, 5),
    # ('DQN Agent', dqn_agent, 5),
], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=2, solve=True, verbose=1, plotting=1,
                                            save=True, file=None)


# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     ('Random', runtime_wrapper(algs_base.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs_base.earliest_release), 1),
#     ('MCTS', partial(algs_timed.mcts, verbose=False), 5),
#     ('DNN Policy', runtime_wrapper(policy_model), 5),
#     # ('DQN Agent', dqn_agent, 5),
# ], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])
#
# runtimes = np.logspace(-2, -1, 20, endpoint=False)
# evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=2, solve=True, verbose=2, plotting=2,
#                             save=False, file=None)

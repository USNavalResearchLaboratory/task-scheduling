from functools import partial
import cProfile
import pstats

import numpy as np
from matplotlib import pyplot as plt

from task_scheduling.util.generic import runtime_wrapper
from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.algorithms import base as algs_base
from task_scheduling.algorithms import runtime as algs_timed
from task_scheduling.learning import environments as envs
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler as SL_Scheduler
from task_scheduling.learning.RL_policy import ReinforcementLearningScheduler as RL_Scheduler


#%%

# NOTE: ensure train/test separation for loaded data, use iter_mode='once'
# NOTE: to train multiple schedulers on same loaded data, use problem_gen.restart(shuffle=False)



# t_release_lim=(0, 0)
# channel_avail_lim = (0, 10)

# problem_gen = problem_gens.Random.relu_drop(n_tasks=5, n_ch=1, rng=None, duration_lim=(0,.05), t_release_lim=(0,0), slope_lim=(0,2), t_drop_lim=(0,6), l_drop_lim=(300,300))
problem_gen = problem_gens.Random.search_track(n_tasks=4, n_ch=1, rng=None, ch_avail_lim=(0, 6))

# problem_gen = problem_gens.Random.relu_drop(n_tasks=3, n_ch=1, rng=None)
# problem_gen = problem_gens.DeterministicTasks.relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.PermutedTasks.relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Dataset.load('relu_c1t8_1000', iter_mode='once', shuffle_mode='once', rng=None)
# problem_gen = problem_gens.Random.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.01))
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2))
# problem_gen = problem_gens.Dataset.load('search_track_c1t8_1000', iter_mode='once', shuffle_mode='once', rng=None)


# Algorithms
features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                     # ('release time', lambda task: task.t_release,
                     #  (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     # ('channel_avail', lambda task: task.slope, problem_gen.ch_avail_gen.lims),
                     ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                     ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                     # ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                     # ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                     # ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])


# sort_func = None
# sort_func = 't_release'
def sort_func(task):
    return task.t_release


weight_func_ = None
# def weight_func_(env):
#     return (env.n_tasks - len(env.node.seq)) / env.n_tasks

env_cls = envs.SeqTasking
# env_cls = envs.StepTasking

env_params = {'node_cls': TreeNodeShift,
              'features': features,
              'sort_func': sort_func,
              'masking': True,
              # 'action_type': 'int',
              'action_type': 'int',
              # 'seq_encoding': 'one-hot', # Not used in SeqTasking
              }

env = env_cls(problem_gen, **env_params)
env.reset()
env.step(1)

# random_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, model_cls='Random', n_episodes=1)
dqn_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params,
                                        model_cls='DQN', model_params={'verbose': 1}, n_episodes=10000,
                                        save=False, save_path=None)

# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')

# policy_model = SL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, layers=None, compile_params=None,
#                                            n_batch_train=90, n_batch_val=10, batch_size=4, weight_func=weight_func_,
#                                            fit_params={'epochs': 100}, do_tensorboard=False, plot_history=True,
#                                            save=False, save_path=None)
# policy_model = SL_Scheduler.load('temp/2020-10-28_14-56-42')


# (tasks, ch_avail), = problem_gen(n_gen=1)

# with cProfile.Profile() as pr:        # TODO: DNN speed vs DQN?
#     out = policy_model(tasks, ch_avail)
#
# # pr.print_stats()
#
# ps = pstats.Stats(pr).sort_stats('cumulative')
# ps.print_stats()
# pr.dump_stats('profile.pstat')


algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    # ('Random', algs_base.random_sequencer, 20),
    ('ERT', algs_base.earliest_release, 1),
    # ('MCTS', partial(algs_base.mcts, n_mc=100, verbose=False), 5),
    ('DQN Agent', dqn_agent, 5),
    # ('DNN Policy', policy_model, 5),
], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=100, solve=True,
                                            verbose=2, plotting=1, save=True, file=None)
# l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=10, solve=True,
#                                             verbose=2, plotting=1, save=True, file=None)

# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     # ('Random', runtime_wrapper(algs_base.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs_base.earliest_release), 1),
#     ('MCTS', partial(algs_timed.mcts, verbose=False), 5),
#     ('DQN Agent', runtime_wrapper(dqn_agent), 5),
#     # ('DNN Policy', runtime_wrapper(policy_model), 5),
# ], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])
#
# runtimes = np.logspace(-2, 0, 30, endpoint=False)
# l_ex_iter = evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=10, solve=True, verbose=3, plotting=1,
#                                         save=False, file=None)
# l_ex_iter = evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=5, solve=True, verbose=3, plotting=1,
#                                         save=False, file=None)
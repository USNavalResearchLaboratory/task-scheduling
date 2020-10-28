from functools import partial

import numpy as np

from task_scheduling.util.results import compare_algorithms
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.tree_search import TreeNodeShift, earliest_release, random_sequencer
from task_scheduling.learning import environments as envs
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler as SL_Scheduler
# from task_scheduling.learning.RL_policy import ReinforcementLearningScheduler as RL_Scheduler


#%%

# NOTE: ensure train/test separation for loaded data, use iter_mode='once'
# NOTE: to train multiple schedulers on same loaded data, use problem_gen.restart(shuffle=False)

# problem_gen = problem_gens.Random.relu_drop(n_tasks=8, n_ch=1)
# problem_gen = problem_gens.DeterministicTasks.relu_drop(n_tasks=8, n_ch=1, rng=None)
problem_gen = problem_gens.PermutedTasks.relu_drop(n_tasks=8, n_ch=1, rng=None)
# problem_gen = problem_gens.Dataset.load('relu_c1t8_1000', iter_mode='once', shuffle_mode='once', rng=None)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., 0.2))
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2))

# Algorithms
features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                     ('release time', lambda task: task.t_release,
                      (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                     ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                     ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                     ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                     ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])


def sort_func(self, n):
    if n in self.node.seq:
        return float('inf')
    else:
        return self.node.tasks[n].t_release


weight_func_ = None
# def weight_func_(env):
#     return (env.n_tasks - len(env.node.seq)) / env.n_tasks

# env_cls = envs.SeqTaskingEnv
env_cls = envs.StepTaskingEnv

env_params = {'node_cls': TreeNodeShift,
              'features': features,
              'sort_func': sort_func,
              'masking': True,
              # 'action_type': 'int',
              'seq_encoding': 'one-hot',
              }

# random_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, model_cls='Random', n_episodes=1)
# dqn_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params,
#                                         model_cls='DQN', model_params=None, n_episodes=1000,
#                                         save=False, save_path=None)
# dqn_agent = RL_Scheduler.load('temp/DQN_2020-10-28_15-44-00', env=None, model_cls='DQN')

policy_model = SL_Scheduler.train_from_gen(problem_gen, env_cls, env_params, layers=None, compile_params=None,
                                           n_batch_train=90, n_batch_val=10, batch_size=4, weight_func=weight_func_,
                                           fit_params={'epochs': 10}, do_tensorboard=False, plot_history=True,
                                           save=True, save_path=None)
# policy_model = SL_Scheduler.load('temp/2020-10-28_14-56-42')

algorithms = np.array([
    # ('B&B', partial(branch_bound, verbose=False), 1),
    # ('MCTS', partial(mcts, n_mc=200, verbose=False), 5),
    ('ERT', partial(earliest_release, do_swap=True), 1),
    ('Random', random_sequencer, 20),
    # ('DQN Agent', dqn_agent, 5),
    ('DNN Policy', policy_model, 5),
], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])

# Compare algorithms
compare_algorithms(algorithms, problem_gen, n_gen=10, solve=True, verbose=2, plotting=1, save=False, file=None)

# for n_tasks, n_ch in ((24, 1), (32, 1),):
#     problem_gen = RandomProblem.relu_drop(n_tasks, n_ch)
#     list(problem_gen(n_gen=1000, solve=True, verbose=True, save=True, file=f'relu_c{n_ch}t{n_tasks}_1000'))
#
#     file = f'relu_c{n_ch}t{n_tasks}_1000'
#     compare_algorithms(algorithms, problem_gen, n_gen=1000,
#                        solve=True, verbose=1, plotting=1, save=True, file=file)


# if __name__ == '__main__':
#     main()

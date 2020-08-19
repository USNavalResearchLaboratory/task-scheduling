"""
Task scheduler comparison.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

from functools import partial

import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt

from util.generic import algorithm_repr
from util.results import check_valid, eval_loss, timing_wrapper
from util.plot import plot_task_losses, scatter_loss_runtime

from generators.scheduling_problems import Random as RandomProblem
from generators.scheduling_problems import Dataset as ProblemDataset

from tree_search import TreeNodeShift, branch_bound, mcts, earliest_release
from env_tasking import StepTaskingEnv, train_agent, load_agent
from SL_policy import train_policy, load_policy

# TODO: structure imports properly w/o relying on PyCharm's path append of the content root

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


# %% Inputs
n_gen = 10      # number of task scheduling problems

solve = True
save = True

problem_gen = RandomProblem.relu_drop_default(n_tasks=4, n_ch=2)
# problem_gen = ProblemDataset.load('temp/2020-08-06_14-15-35', iter_mode='repeat', shuffle=True, rng=None)

# FIXME: ensure train/test separation for loaded problem data
# TODO: save option? buffer data during multiple gen calls?

# Algorithms

features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                     ('release time', lambda task: task.t_release, (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                     ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                     ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                     ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                     ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

# features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.duration_lim),
#                      ('release time', lambda task: task.t_release, (0., problem_gen.task_gen.t_release_lim[1])),
#                      ('slope', lambda task: task.slope, problem_gen.task_gen.slope_lim),
#                      ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.t_drop_lim[1])),
#                      ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.l_drop_lim[1])),
#                      ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
#                      ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
#                      ],
#                     dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])


def sort_func(self, n):
    if n in self.node.seq:
        return float('inf')
    else:
        return self.node.tasks[n].t_release


env_cls = StepTaskingEnv
env_params = {'node_cls': TreeNodeShift,
              'features': features,
              'sort_func': sort_func,
              'seq_encoding': 'indicator'
              }

agent_file = None
# agent_file = 'temp/2020-08-06_14-15-31'

if agent_file is None:
    random_agent = train_agent(problem_gen,
                               n_gen_train=3, n_gen_val=2, env_cls=env_cls, env_params=env_params,
                               save=True, save_dir=None)
elif type(agent_file) == str:
    random_agent = load_agent(agent_file)
else:
    raise ValueError("Parameter 'agent_file' must be string or None.")

# model_file = None
# # model_file = 'temp/2020-08-03_12-52-22'
#
# if model_file is None:
#     network_policy = train_policy(problem_gen,
#                                   n_gen_train=10, n_gen_val=10, env_cls=env_cls, env_params=env_params,
#                                   model=None, compile_params=None, fit_params=None,
#                                   do_tensorboard=False, plot_history=True, save=True, save_dir=None)
# elif type(model_file) == str:
#     network_policy = load_policy(model_file)
# else:
#     raise ValueError("Parameter 'agent_file' must be string or None.")


algorithms = np.array([
    # ('B&B', partial(branch_bound, verbose=False), 1),
    ('MCTS', partial(mcts, n_mc=200, verbose=False), 5),
    ('ERT', partial(earliest_release, do_swap=True), 1),
    ('Random Agent', partial(random_agent), 20),
    # ('DNN Policy', partial(network_policy), 1),
                     ], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])


# %% Evaluate

if solve:
    _args_iter = {'object': [([np.nan],) + tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen,
                  'dtype': [('B&B Optimal', np.float, (1,))] + [(alg['name'], np.float, (alg['n_iter'],))
                                                                for alg in algorithms]}
    _args_mean = {'object': [(np.nan,) * (1 + len(algorithms))] * n_gen,
                  'dtype': [('B&B Optimal', np.float)] + [(alg['name'], np.float) for alg in algorithms]}
else:
    _args_iter = {'object': [tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen,
                  'dtype': [(alg['name'], np.float, (alg['n_iter'],)) for alg in algorithms]}
    _args_mean = {'object': [(np.nan,) * len(algorithms)] * n_gen,
                  'dtype': [(alg['name'], np.float) for alg in algorithms]}

t_run_iter = np.array(**_args_iter)
l_ex_iter = np.array(**_args_iter)

t_run_mean = np.array(**_args_mean)
l_ex_mean = np.array(**_args_mean)

for i_gen, out_gen in enumerate(problem_gen(n_gen, solve=solve, save=save)):      # Generate new scheduling problem
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    if solve:
        (tasks, ch_avail), (t_ex_opt, ch_ex_opt, t_run_opt) = out_gen

        check_valid(tasks, t_ex_opt, ch_ex_opt)
        l_ex_opt = eval_loss(tasks, t_ex_opt)

        t_run_iter['B&B Optimal'][i_gen, 0] = t_run_opt
        l_ex_iter['B&B Optimal'][i_gen, 0] = l_ex_opt
        t_run_mean['B&B Optimal'][i_gen] = t_run_opt
        l_ex_mean['B&B Optimal'][i_gen] = l_ex_opt

        print(f'  B&B Optimal', end='\r')
        print('')
        print(f"    Avg. Runtime: {t_run_mean['B&B Optimal'][i_gen]:.2f} (s)")
        print(f"    Avg. Execution Loss: {l_ex_mean['B&B Optimal'][i_gen]:.2f}")

    else:
        tasks, ch_avail = out_gen

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_iter in algorithms:
        for iter_ in range(n_iter):      # Perform new algorithm runs
            print(f'  {alg_repr} - Iteration: {iter_ + 1}/{n_iter}', end='\r')

            # Run algorithm
            t_ex, ch_ex, t_run = timing_wrapper(alg_func)(tasks, ch_avail)

            # Evaluate schedule
            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            # Store loss and runtime
            t_run_iter[alg_repr][i_gen, iter_] = t_run
            l_ex_iter[alg_repr][i_gen, iter_] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

        t_run_mean[alg_repr][i_gen] = t_run_iter[alg_repr][i_gen].mean()
        l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean()

        print('')
        print(f"    Avg. Runtime: {t_run_mean[alg_repr][i_gen]:.2f} (s)")
        print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][i_gen]:.2f}")

    scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])


# %% Results
_, ax_results = plt.subplots(num='Results', clear=True)
scatter_loss_runtime(t_run_mean, l_ex_mean,
                     ax=ax_results, ax_kwargs={'title': f'Performance on random sets of {problem_gen.n_tasks} tasks'})

print('\nAvg. Performance\n' + 16*'-')
print(f"{'Algorithm:':<35}{'Loss:':<8}{'Runtime (s):':<10}")
if solve:
    print(f"{'B&B Optimal':<35}{l_ex_mean['B&B Optimal'].mean():<8.2f}{t_run_mean['B&B Optimal'].mean():<10.6f}")
for rep in algorithms['name']:
    print(f"{rep:<35}{l_ex_mean[rep].mean():<8.2f}{t_run_mean[rep].mean():<10.6f}")


# Relative to B&B
if solve:
    t_run_mean_opt = t_run_mean['B&B Optimal'].copy()
    l_ex_mean_opt = l_ex_mean['B&B Optimal'].copy()

    t_run_mean_norm = t_run_mean.copy()
    l_ex_mean_norm = l_ex_mean.copy()

    # t_run_mean_norm['B&B Optimal'] = 0.
    l_ex_mean_norm['B&B Optimal'] = 0.
    for rep in algorithms['name']:
        # t_run_mean_norm[rep] -= t_run_mean_opt
        # t_run_mean_norm[rep] /= t_run_mean_opt
        l_ex_mean_norm[rep] -= l_ex_mean_opt
        l_ex_mean_norm[rep] /= l_ex_mean_opt

    _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
    scatter_loss_runtime(t_run_mean_norm, l_ex_mean_norm,
                         ax=ax_results_norm,
                         ax_kwargs={'title': f'Relative Performance on random sets of {problem_gen.n_tasks} tasks',
                                    'ylabel': 'Excess Loss (Normalized)',
                                    # 'xlabel': 'Runtime Difference (Normalized)'
                                    }
                         )

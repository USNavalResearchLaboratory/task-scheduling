"""
Task scheduler comparison.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

from time import perf_counter
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from util.generic import algorithm_repr
from util.results import check_valid, eval_loss
from util.plot import plot_task_losses, scatter_loss_runtime

from generators.scheduling_problems import Random as RandomProblem
from generators.scheduling_problems import Load as LoadProblem

from tree_search import TreeNodeShift, branch_bound, mcts, earliest_release
from env_tasking import StepTaskingEnv, train_agent, load_agent
from SL_policy import train_policy, load_policy

# TODO: structure imports properly w/o relying on PyCharm's path append of the content root

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


# %% Inputs
n_gen = 2      # number of task scheduling problems

problem_gen = RandomProblem.relu_drop_default(n_tasks=4, n_ch=2)
# problem_gen = LoadProblem('temp/2020-07-30_14-11-07')


# TODO: add functionality for loading problems and B&B solutions


# Algorithms

features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.duration_lim),
                     ('release time', lambda task: task.t_release, (0., problem_gen.task_gen.t_release_lim[1])),
                     ('slope', lambda task: task.slope, problem_gen.task_gen.slope_lim),
                     ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.t_drop_lim[1])),
                     ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.l_drop_lim[1])),
                     ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                     ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])


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
# agent_file = 'temp/2020-07-24_14-31-45'

if agent_file is None:
    random_agent = train_agent(problem_gen,
                               # problem_gen.n_tasks, problem_gen.task_gen, problem_gen.n_ch, problem_gen.ch_avail_gen,
                               n_gen_train=0, n_gen_val=0, env_cls=env_cls, env_params=env_params,
                               save=True, save_dir=None)
elif type(agent_file) == str:
    random_agent = load_agent(agent_file)
else:
    raise ValueError("Parameter 'agent_file' must be string or None.")


policy_file = None
# policy_file = 'temp/2020-07-23_13-09-17'

if policy_file is None:
    network_policy = train_policy(problem_gen,
                                  # problem_gen.n_tasks, problem_gen.task_gen, problem_gen.n_ch, problem_gen.ch_avail_gen,
                                  n_gen_train=10, n_gen_val=10, env_cls=env_cls, env_params=env_params,
                                  model=None, compile_params=None, fit_params=None,
                                  do_tensorboard=False, plot_history=True, save=True, save_dir=None)
elif type(policy_file) == str:
    network_policy = load_policy(policy_file)
else:
    raise ValueError("Parameter 'agent_file' must be string or None.")


alg_funcs = [
    partial(branch_bound, verbose=False),
    partial(mcts, n_mc=200, verbose=False),
    partial(earliest_release, do_swap=True),
    partial(random_agent),
    partial(network_policy),
             ]

alg_n_iter = [1, 5, 1, 50, 1]       # number of runs per problem
alg_reprs = list(map(algorithm_repr, alg_funcs))    # string representations


# %% Evaluate
t_run_iter = np.array(list(zip(*[np.empty((n_gen, n_iter)) for n_iter in alg_n_iter])),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_iter])))

l_ex_iter = np.array(list(zip(*[np.empty((n_gen, n_iter)) for n_iter in alg_n_iter])),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_iter])))

t_run_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

for i_gen, (tasks, ch_avail) in enumerate(problem_gen(n_gen)):      # Generate new scheduling problem
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_iter in zip(alg_reprs, alg_funcs, alg_n_iter):
        for iter_ in range(n_iter):      # Perform new algorithm runs
            print(f'  {alg_repr} - Iteration: {iter_ + 1}/{n_iter}', end='\r')

            t_start = perf_counter()
            t_ex, ch_ex = alg_func(tasks, ch_avail)
            t_run = perf_counter() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

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
for rep in alg_reprs:
    print(f"{rep:<35}{l_ex_mean[rep].mean():<8.2f}{t_run_mean[rep].mean():<10.6f}")


# Relative to B&B
if 'branch_bound' in alg_reprs:
    t_run_mean_bb = t_run_mean['branch_bound'].copy()
    l_ex_mean_bb = l_ex_mean['branch_bound'].copy()

    t_run_mean_norm = t_run_mean.copy()
    l_ex_mean_norm = l_ex_mean.copy()
    for rep in alg_reprs:
        # t_run_mean_norm[rep] -= t_run_mean_bb
        # t_run_mean_norm[rep] /= t_run_mean_bb
        l_ex_mean_norm[rep] -= l_ex_mean_bb
        l_ex_mean_norm[rep] /= l_ex_mean_bb

    _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
    scatter_loss_runtime(t_run_mean_norm, l_ex_mean_norm,
                         ax=ax_results_norm,
                         ax_kwargs={'title': f'Relative Performance on random sets of {problem_gen.n_tasks} tasks',
                                    'ylabel': 'Excess Loss (Normalized)',
                                    # 'xlabel': 'Runtime Difference (Normalized)'
                                    }
                         )

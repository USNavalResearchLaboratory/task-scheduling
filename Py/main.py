"""
Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

# TODO: add proper main() def, __name__ conditional execution?

import time     # TODO: use builtin module timeit instead? or cProfile?
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tasks import ReluDropGenerator
from tree_search import branch_bound, mc_tree_search, random_sequencer, earliest_release, est_alg_kw
from util.generic import algorithm_repr
from util.results import check_valid, eval_loss
from util.plot import plot_task_losses, plot_schedule, plot_results

plt.style.use('seaborn')


# %% Inputs

n_gen = 2      # number of task scheduling problems
n_run = 5       # number of runs per problem

ch_avail = np.zeros(2)     # channel availability times

task_gen = partial(ReluDropGenerator().rand_tasks, n_tasks=8)

# Algorithms
alg_funcs = [partial(branch_bound, ch_avail=ch_avail, verbose=False),
             partial(mc_tree_search, ch_avail=ch_avail, n_mc=100, verbose=False),
             partial(earliest_release, ch_avail=ch_avail, do_swap=True),
             partial(random_sequencer, ch_avail=ch_avail)]

alg_n_runs = [5, 5, 1, 5]

alg_reprs = list(map(algorithm_repr, alg_funcs))


# %% Evaluate
t_run_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

l_ex_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

t_run_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

for i_gen in range(n_gen):      # Generate new tasks
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    tasks = task_gen()

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
        for i_run in range(n_run):      # Perform new algorithm runs
            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex = alg_func(tasks)
            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            t_run_iter[alg_repr][i_gen, i_run] = t_run
            l_ex_iter[alg_repr][i_gen, i_run] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

        t_run_mean[alg_repr][i_gen] = t_run_iter[alg_repr][i_gen].mean()
        l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean()

        print('')
        print(f"    Avg. Runtime: {t_run_mean[alg_repr][i_gen]:.2f} seconds")
        print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][i_gen]:.3f}")

    plot_results(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

plot_results(t_run_mean, l_ex_mean, ax_kwargs={'title': 'Average performance on random task sets'})

print('')

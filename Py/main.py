"""Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.
"""

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
algorithms = [partial(branch_bound, ch_avail=ch_avail, verbose=False),
              partial(mc_tree_search, ch_avail=ch_avail, n_mc=100, verbose=False),
              partial(earliest_release, ch_avail=ch_avail, do_swap=True),
              partial(random_sequencer, ch_avail=ch_avail)]


# %% Evaluate
t_run_iter = np.empty((n_gen, len(algorithms), n_run))
l_ex_iter = np.empty((n_gen, len(algorithms), n_run))

t_run_mean = np.empty((len(algorithms), n_gen))
l_ex_mean = np.empty((len(algorithms), n_gen))

# _, ax_tasks = plt.subplots(1, n_gen, num='Task Loss Functions', clear=True)
alg_reprs = list(map(algorithm_repr, algorithms))

for i_gen in range(n_gen):      # Generate new tasks
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    tasks = task_gen()

    # plot_task_losses(tasks, ax=ax_tasks[i_gen])
    _, ax_res = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_res[0])

    for i_alg, alg in enumerate(algorithms):

        for i_run in range(n_run):      # Perform new algorithm runs
            print(f'  ({i_alg + 1}/{len(algorithms)}) {alg_reprs[i_alg]}, Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex = alg(tasks)
            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            t_run_iter[i_gen, i_alg, i_run] = t_run
            l_ex_iter[i_gen, i_alg, i_run] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_str=alg_reprs[i_alg], ax=None)

        t_run_mean[i_alg, i_gen] = t_run_iter[i_gen, i_alg].mean()
        l_ex_mean[i_alg, i_gen] = l_ex_iter[i_gen, i_alg].mean()
        # print('')
        # print(f"    Avg. Runtime: {t_run_iter[i_gen, i_alg].mean():.2f} seconds")
        # print(f"    Avg. Execution Loss: {l_ex_iter[i_gen, i_alg].mean():.3f}")

    plot_results(alg_reprs, t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_res[1])

plot_results(alg_reprs, t_run_mean, l_ex_mean)
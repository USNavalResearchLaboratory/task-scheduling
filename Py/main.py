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
n_run = 2       # number of runs per problem

ch_avail = np.zeros(2)     # channel availability times

task_gen = partial(ReluDropGenerator().rand_tasks, n_tasks=8)

# Algorithms
algorithms = [partial(branch_bound, ch_avail=ch_avail, verbose=True),
              partial(mc_tree_search, ch_avail=ch_avail, n_mc=100, verbose=True),
              partial(earliest_release, ch_avail=ch_avail, do_swap=True),
              partial(random_sequencer, ch_avail=ch_avail)]

# %% Evaluate
t_run_alg = np.empty((n_gen, len(algorithms), n_run))
l_ex_alg = np.empty((n_gen, len(algorithms), n_run))

# _, ax_tasks = plt.subplots(1, n_gen, num='Task Loss Functions', clear=True)
alg_reprs = list(map(algorithm_repr, algorithms))

for i_gen in range(n_gen):      # Generate new tasks

    tasks = task_gen()

    # plot_task_losses(tasks, ax=ax_tasks[i_gen])
    _, ax_res = plt.subplots(2, 1, num=f'Task Set: {i_gen}', clear=True)
    plot_task_losses(tasks, ax=ax_res[0])

    for i_alg, alg in enumerate(algorithms):
        # print(f'\nAlgorithm: {alg_reprs[i_alg]}')       # TODO: print run logs

        for i_run in range(n_run):      # Perform new algorithm runs
            t_start = time.time()
            t_ex, ch_ex = alg(tasks)
            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            t_run_alg[i_gen, i_alg, i_run] = t_run
            l_ex_alg[i_gen, i_alg, i_run] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_str=alg_reprs[i_alg], ax=None)


            # print("Task Execution Channels: " + ", ".join([f'{ch}' for ch in ch_ex]))     # TODO: print funcs?
            # print("Task Execution Times: " + ", ".join([f'{t:.3f}' for t in t_ex]))
            # print(f"Execution Loss: {l_ex:.3f}")
            # print(f"Runtime: {t_run:.2f} seconds")

    plot_results(alg_reprs, t_run_alg[i_gen], l_ex_alg[i_gen], ax=ax_res[1])

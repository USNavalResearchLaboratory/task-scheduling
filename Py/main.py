"""Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.
"""

import time     # TODO: use builtin module timeit instead? or cProfile?
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tasks import ReluDropGenerator
from tree_search import branch_bound, mc_tree_search, random_sequencer, EstAlg, est_alg
from util.utils import check_valid, eval_loss, plot_task_losses

plt.style.use('seaborn')


# %% Inputs

n_gen = 2      # number of task scheduling problems
n_run = 1       # number of runs per problem

ch_avail = np.zeros(2)     # channel availability times

task_gen = partial(ReluDropGenerator().rand_tasks, n_tasks=8)

# Algorithms

algorithms = [partial(branch_bound, ch_avail=ch_avail, verbose=True),
              partial(mc_tree_search, ch_avail=ch_avail, n_mc=100, verbose=True),
              partial(est_alg, ch_avail=ch_avail),
              partial(random_sequencer, ch_avail=ch_avail)]


# %% Evaluate

# t_ex_alg = np.empty((n_gen, len(algorithms), n_run, n_tasks))
# ch_ex_alg = np.empty((n_gen, len(algorithms), n_run, n_tasks))

t_run_alg = np.empty((n_gen, len(algorithms), n_run))
l_ex_alg = np.empty((n_gen, len(algorithms), n_run))

_, ax_tasks = plt.subplots(1, n_gen, num='Task Loss Functions', clear=True)

for i_gen in range(n_gen):      # Generate new tasks

    tasks = task_gen()
    plot_task_losses(tasks, ax=ax_tasks[i_gen])

    for i_alg, alg in enumerate(algorithms):
        print(f'\nAlgorithm: {alg.func.__name__}')

        for i_run in range(n_run):      # Perform new algorithm runs
            t_start = time.time()
            t_ex, ch_ex = alg(tasks)
            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            # t_ex_alg[i_gen, i_alg, i_run] = t_ex
            # ch_ex_alg[i_gen, i_alg, i_run] = ch_ex

            t_run_alg[i_gen, i_alg, i_run] = t_run
            l_ex_alg[i_gen, i_alg, i_run] = l_ex

            # # Results
            # print('')
            # print("Task Execution Channels: " + ", ".join([f'{ch}' for ch in ch_ex]))
            # print("Task Execution Times: " + ", ".join([f'{t:.3f}' for t in t_ex]))
            # print(f"Execution Loss: {l_ex:.3f}")
            # print(f"Runtime: {t_run:.2f} seconds")





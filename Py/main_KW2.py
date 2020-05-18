"""Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.
"""

import time     # TODO: use builtin module timeit instead?
# TODO: do cProfile!
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tasks import ReluDropGenerator
from tree_search import branch_bound, mc_tree_search, random_sequencer, est_alg, est_task_swap_alg, ed_alg, ed_swap_task_alg
from util.results import check_valid, eval_loss

plt.style.use('seaborn')

rng = np.random.default_rng(100)


# %% Inputs

n_gen = 1      # number of task scheduling problems
n_run = 2       # number of runs per problem

ch_avail = np.zeros(2)     # channel availability times

n_tasks = 8      # number of tasks
task_gen = partial(ReluDropGenerator(rng).rand_tasks, n_tasks)

# Algorithms
algorithms = [partial(est_task_swap_alg, ch_avail=ch_avail),
              partial(est_alg, ch_avail=ch_avail),
              partial(ed_alg, ch_avail=ch_avail),
              partial(ed_swap_task_alg, ch_avail=ch_avail),
              partial(branch_bound, ch_avail=ch_avail, verbose=True, rng=rng),
              partial(mc_tree_search, ch_avail=ch_avail, n_mc=1000, verbose=True, rng=rng),
              partial(random_sequencer, ch_avail=ch_avail, rng=rng)]


# %% Evaluate

# t_ex_alg = np.empty((n_gen, len(algorithms), n_run, n_tasks))
# ch_ex_alg = np.empty((n_gen, len(algorithms), n_run, n_tasks))

t_run_alg = np.empty((n_gen, len(algorithms), n_run))
l_ex_alg = np.empty((n_gen, len(algorithms), n_run))

for i_gen in range(n_gen):      # Generate new tasks

    tasks = task_gen()

    for i_alg, alg in enumerate(algorithms):
        print(f'\nAlgorithm: {alg.func.__name__} \n')

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



        # Results
        print('')
        # print("Task Execution Channels: " + ", ".join([f'{ch}' for ch in ch_ex]))
        # print("Task Execution Times: " + ", ".join([f'{t:.3f}' for t in t_ex]))
        print(f"Execution Loss: {l_ex:.3f}")
        print(f"Runtime: {t_run:.2f} seconds")







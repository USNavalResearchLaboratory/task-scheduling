"""Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.
"""

import time     # TODO: use builtin module timeit instead?
# TODO: do cProfile!
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tasks import ReluDropGenerator
from tree_search import branch_bound, mc_tree_search, random_sequencer
from util.utils import check_valid, eval_loss

plt.style.use('seaborn')

rng = np.random.default_rng()


# %% Inputs

n_run = 10      # number of task scheduling problems

# Setup
ch_avail = np.zeros(2)     # channel availability times

n_tasks = 8      # number of tasks
task_gen = ReluDropGenerator(rng)

# Algorithms
algorithms = [partial(branch_bound, ch_avail=ch_avail, verbose=True, rng=rng),
              partial(mc_tree_search, ch_avail=ch_avail, n_mc=1000, verbose=True, rng=rng),
              partial(random_sequencer, ch_avail=ch_avail, rng=rng)]


# %% Evaluate
t_ex_alg = np.empty((n_run, len(algorithms)))
ch_ex_alg = np.empty((n_run, len(algorithms)))
l_ex_alg = np.empty((n_run, len(algorithms)))
t_run_alg = np.empty((n_run, len(algorithms)))
for i_run in range(n_run):

    tasks = task_gen.rand_tasks(n_tasks)

    for alg in algorithms:
        print(f'\nAlgorithm: {alg.func.__name__} \n')       # TODO: logging?

        t_start = time.time()
        t_ex, ch_ex = alg(tasks)
        t_run = time.time() - t_start

        check_valid(tasks, t_ex, ch_ex)
        l_ex = eval_loss(tasks, t_ex)

        t_ex_alg[i_run, i_alg]
        ch_ex_alg.append(ch_ex)
        t_run_alg.append(t_run)
        l_ex_alg.append(l_ex)

        # Results
        print('')
        print("Task Execution Channels: " + ", ".join([f'{ch}' for ch in ch_ex]))
        print("Task Execution Times: " + ", ".join([f'{t:.3f}' for t in t_ex]))
        print(f"Execution Loss: {l_ex:.3f}")
        print(f"Runtime: {t_run:.2f} seconds")







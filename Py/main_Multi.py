"""
Task scheduling.
"""

import time     # TODO: use builtin module timeit instead?
# TODO: do cProfile!
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from task_obj import TasksRRM
from Tree_Search_Multi import branch_bound, mc_tree_search

plt.style.use('seaborn')

rng = np.random.default_rng()
# rng = np.random.RandomState(100)


# %% Inputs

n_channels = 2       # number of channels

# Tasks
n_tasks = 8      # number of tasks

t_release = rng.uniform(0, 10, n_tasks)

duration = rng.uniform(1, 3, n_tasks)

w = rng.uniform(0.8, 1.2, n_tasks)
t_drop = t_release + duration * rng.uniform(3, 5, n_tasks)
l_drop = rng.uniform(2, 3, n_tasks) * w * (t_drop - t_release)

tasks = []
for n in range(n_tasks):
    tasks.append(TasksRRM.lin_drop(t_release[n], duration[n], w[n], t_drop[n], l_drop[n]))

del duration, t_release, w, t_drop, l_drop


# Algorithms
algorithms = [partial(branch_bound, n_ch=n_channels, verbose=True, rng=rng),
              partial(mc_tree_search, n_ch=n_channels, n_mc=1000, verbose=True, rng=rng)]


# %% Evaluate
t_ex_alg, ch_ex_alg, l_ex_alg, t_run_alg = [], [], [], []
for alg in algorithms:
    print(f'\nAlgorithm: {alg.func} \n')

    tic = time.time()
    t_ex, ch_ex = alg(tasks)
    t_run = time.time() - tic

    t_ex_alg.append(t_ex)
    ch_ex_alg.append(ch_ex)
    t_run_alg.append(t_run)

    # Check solution validity
    for ch in range(n_channels):
        tasks_ch = np.asarray(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch) - 1):
            for n_2 in range(n_1 + 1, len(tasks_ch)):
                if t_ex_ch[n_1] - tasks_ch[n_2].duration + 1e-12 < t_ex_ch[n_2] < t_ex_ch[n_1] \
                        + tasks_ch[n_1].duration - 1e-12:
                    raise ValueError('Invalid Solution: Scheduling Conflict')

    # Cost evaluation
    l_ex = 0
    for n in range(n_tasks):
        l_ex += tasks[n].loss_fcn(t_ex[n])
    l_ex_alg.append(l_ex)

    # Results
    print('')
    print("Task Execution Channels: " + ", ".join([f'{ch}' for ch in ch_ex]))
    print("Task Execution Times: " + ", ".join([f'{t:.3f}' for t in t_ex]))
    print(f"Achieved Loss: {l_ex:.3f}")
    print(f"Runtime: {t_run:.2f} seconds")


# %% Plots
t_plot_max = 0
for t_ex in t_ex_alg:
    t_plot_max = max(t_plot_max, max(t_ex))
t_plot_max += max([t.duration for t in tasks])

t_plot = np.arange(0, t_plot_max, 0.01)

plt.figure(num='Task Loss Functions', clear=True)
for n in range(n_tasks):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
plt.gca().set(xlabel='t', ylabel='Loss')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(t_plot[[0, -1]])
plt.grid(True)
plt.legend()


bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(algorithms)):
    plt.figure(num=str(algorithms[i].func.__name__), clear=True, figsize=[8, 2.5])
    plt.title(f'Loss = {l_ex_alg[i]:.3f}')
    # d = ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    for n in range(len(tasks)):
        plt.gca().broken_barh([(t_ex_alg[i][n], tasks[n].duration)], (ch_ex_alg[i][n]-0.5, 1),
                              facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=f'Task #{n}')

    plt.gca().set(xlim=t_plot[[0, -1]], ylim=(-.5, n_channels-1+.5),
                  xlabel='t', yticks=list(range(n_channels)), ylabel='Channel')
    plt.gca().grid(True)
    plt.gca().legend()

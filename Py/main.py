"""
Task scheduling.
"""

import time     # TODO: use builtin module timeit instead?

import numpy as np
import matplotlib.pyplot as plt

from task_obj import TasksRRM
from Tree_Search import branch_bound, mc_tree_search
# from BranchBound_original import branch_bound

plt.style.use('seaborn')

# rng = np.random.default_rng()
rng = np.random.RandomState(100)


# %% Inputs

# Tasks
N = 10      # number of tasks

t_start = rng.uniform(0, 30, N)
duration = rng.uniform(1, 3, N)

w = rng.uniform(0.8, 1.2, N)
t_drop = t_start + duration * rng.uniform(3, 5, N)
l_drop = rng.uniform(2, 3, N) * w * (t_drop - t_start)

tasks = []
for n in range(N):
    tasks.append(TasksRRM.lin_drop(t_start[n], duration[n], w[n], t_drop[n], l_drop[n]))

del duration, t_start, w, t_drop, l_drop


# %% Branch and Bound
tic = time.time()
# t_ex, loss_ex = branch_bound(tasks, verbose=True)
t_ex, loss_ex = mc_tree_search(tasks, N_mc=10000, verbose=True)
t_run = time.time() - tic


# %% Results
print("Task Execution Times: " + ", ".join([f'{t:.2f}' for t in t_ex]))
print(f"Achieved Loss: {loss_ex:.2f}")
print(f"Runtime: {t_run:.2f} seconds")

# Cost evaluation
l_calc = 0
for n in range(N):
    l_calc += tasks[n].loss_fcn(t_ex[n])
if abs(l_calc - loss_ex) > 1e-12:
    raise ValueError('Iterated loss is inaccurate')

# Check solution validity
for n_1 in range(N-1):
    for n_2 in range(n_1+1, N):
        if t_ex[n_1] - tasks[n_2].duration + 1e-12 < t_ex[n_2] < t_ex[n_1] + tasks[n_1].duration - 1e-12:
            raise ValueError('Invalid Solution: Scheduling Conflict')



# %% Plots
t_plot = np.arange(0, max(t_ex) + max([t.duration for t in tasks]), 0.01)
plt.figure(num='Task Loss Functions', clear=True)
for n in range(N):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
plt.gca().set(xlabel='t', ylabel='Loss')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(t_plot[[0, -1]])
plt.grid(True)
plt.legend()


bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(num='Task Schedule', clear=True, figsize=[8, 2.5])
# d = ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
for n in range(len(tasks)):
    plt.gca().broken_barh([(t_ex[n], tasks[n].duration)], (-0.5, 1), facecolors=bar_colors[n % len(bar_colors)], label=f'Task #{n}')

plt.gca().set(xlim=t_plot[[0, -1]], ylim=(-.6, .6), xlabel='t', yticks=[0], yticklabels=['Timeline #1'])
plt.gca().grid(True)
plt.gca().legend()

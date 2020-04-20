"""
Task scheduling.
"""

import time     # TODO: use builtin module timeit instead?

import numpy as np
import matplotlib.pyplot as plt

from task_obj import TasksRRM
from BranchBound import branch_bound

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

t_plot = np.arange(0, np.ceil(t_drop.max()), 0.01)
plt.figure(num='tasks', clear=True)
for n in range(N):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
    plt.gca().set(title='Task Losses', xlabel='t', ylabel='Loss')
    plt.gca()
    plt.grid(True)
    plt.legend()

del duration, t_start, w, t_drop, l_drop


# %% Branch and Bound
tic = time.time()
t_ex, loss_ex = branch_bound(tasks, verbose=False)
t_run = time.time() - tic


# %% Results
print(f"Task Execution Times: {t_ex}")      # TODO: unpack list elements and format?
print(f"Achieved Loss: {loss_ex:.2f}")
print(f"Runtime: {t_run:.2f} seconds")

# Cost evaluation
l_calc = 0
for n in range(N):
    l_calc += tasks[n].loss_fcn(t_ex[n])
if abs(l_calc - loss_ex) > 1e-12:
    raise ValueError('Iterated loss is inaccurate')

# Check solution validity
valid = True
for n_1 in range(N-1):
    for n_2 in range(n_1+1, N):
        cond_1 = t_ex[n_1] >= (t_ex[n_2] + tasks[n_2].duration)
        cond_2 = t_ex[n_2] >= (t_ex[n_1] + tasks[n_1].duration)
        valid = valid and (cond_1 or cond_2)
        if not valid:
            raise ValueError('Invalid Solution: Scheduling Conflict')

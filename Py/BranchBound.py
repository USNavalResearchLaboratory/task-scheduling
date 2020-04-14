"""
Branch and Bound simulation example...
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import random

# from functools import partial

from task_loss_fcns import TasksRRM
from branch_update import branch_update

# rng = np.random.default_rng()
rng = np.random.RandomState(100)

# %% Inputs

# %% Algorithm

def stack_queue(s, b): return [b] + s     # LIFO

# Tasks
N = 10  # number of tasks

t_start = 30 * rng.random(N)
duration = 1 + 2 * rng.random(N)

w = 0.8 + 0.4 * rng.random(N)
t_drop = t_start + duration * (3 + 2 * rng.random(N))
l_drop = (2 + rng.random(N)) * w * (t_drop - t_start)

tasks = []
for n in range(N):
    tasks.append(TasksRRM.lin_drop(t_start[n], duration[n], w[n], t_drop[n], l_drop[n]))


t_plot = np.arange(0, np.ceil(max(t_drop)), 0.01)
plt.figure(num='tasks', clear=True)
for n in range(N):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
    plt.gca().set(title='Task Losses', xlabel='t', ylabel='Loss')
    # plt.grid(True)
    plt.legend()

# del duration, t_start, w, t_drop, l_drop


# %% Tree Search

# TODO: add tic

# Initialize Stack
LB = 0
UB = 0

# t_s_max = t_start.max() + duration.sum()
t_s_max = max([task.t_start for task in tasks]) + sum([task.duration for task in tasks])
for n in range(N):
    LB += tasks[n].loss_fcn(tasks[n].t_start)
    UB += tasks[n].loss_fcn(t_s_max)

# TODO: research numpy array use for named fields like 'struct'
# S0 = np.array([(3, tuple(np.repeat(np.inf, N)), 0, LB, UB)],
#               dtype=[('seq', 'int32'), ('t_ex', 'float64'), ('l_inc', 'float64'), ('LB', 'float64'), ('UB', 'float64')])

S = [{'seq': [], 't_ex': np.full(N, np.inf), 'l_inc': 0, 'LB': LB, 'UB': UB}]

# Iterate
while (len(S) != 1) or (len(S[0]['seq']) != N):
    print(f'# Remaining Branches = {len(S)}')       # TODO: use end=r?

    # Extract Branch
    for i in range(len(S)):
        if len(S[i]['seq']) != N:
            B = S.pop(i)
            break

    # Split Branch
    # T_c = np.setdiff1d(np.arange(N), B['seq'])
    T_c = list(set(range(N)) - set(B['seq']))
    seq_rem = T_c
    # seq_rem = rng.permutation(T_c)
    for n in seq_rem:
        # Generate new branch
        B_new = branch_update(B, n, tasks)

        # Cut Branches
        if B_new['LB'] >= min([B['UB'] for B in S]+[np.inf]):
            None
            # New branch is dominated
        else:
            # Cut Dominated Branches
            S = [br for br in S if br['LB'] < B_new['UB']]

            # Add New Branch to Stack
            S = stack_queue(S, B_new)


l_opt = S[0]['l_inc']
t_ex_opt = S[0]['t_ex']

# t_run = toc
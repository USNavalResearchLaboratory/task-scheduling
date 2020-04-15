"""
Branch and Bound simulation example.
"""

import time     # TODO: use builtin module timeit instead?

import numpy as np
import matplotlib.pyplot as plt

from numpy import random

from task_obj import TasksRRM
from branch_update import branch_update

rng = np.random.default_rng()
# rng = np.random.RandomState(100)

# %% Inputs

# Algorithm
def stack_queue(s, b): return [b] + s     # LIFO

# Tasks
N = 10      # number of tasks

t_start = 30 * rng.random(N)
duration = 1 + 2 * rng.random(N)

w = 0.8 + 0.4 * rng.random(N)
t_drop = t_start + duration * (3 + 2 * rng.random(N))
l_drop = (2 + rng.random(N)) * w * (t_drop - t_start)

tasks = []
for n in range(N):
    tasks.append(TasksRRM.lin_drop(t_start[n], duration[n], w[n], t_drop[n], l_drop[n]))

del duration, t_start, w, t_drop, l_drop


# %% Tree Search

tic = time.time()

# Initialize Stack
LB = 0
UB = 0

t_s_max = max([task.t_start for task in tasks]) + sum([task.duration for task in tasks])
for n in range(N):
    LB += tasks[n].loss_fcn(tasks[n].t_start)
    UB += tasks[n].loss_fcn(t_s_max)

S = [{'seq': [], 't_ex': np.full(N, np.inf), 'l_inc': 0, 'LB': LB, 'UB': UB}]

# Iterate
while (len(S) != 1) or (len(S[0]['seq']) != N):
    print(f'# Remaining Branches = {len(S)}', end='\n')

    # Extract Branch
    for i in range(len(S)):
        if len(S[i]['seq']) != N:
            B = S.pop(i)
            break

    # Split Branch
    T_c = set(range(N)) - set(B['seq'])
    seq_rem = rng.permutation(list(T_c))
    for n in seq_rem:
        B_new = branch_update(B, n, tasks)      # Generate new branch

        if B_new['LB'] < min([br['UB'] for br in S] + [np.inf]):        # New branch is not dominated
            S = [br for br in S if br['LB'] < B_new['UB']]      # Cut Dominated Branches
            S = stack_queue(S, B_new)       # Add New Branch to Stack

l_opt = S[0]['l_inc']
t_ex_opt = S[0]['t_ex']

t_run = time.time() - tic


# %% Results
print(l_opt)
print(t_ex_opt)

if len(S) != 1:
    raise ValueError('Multiple leafs...')

if not all([s['LB'] == s['UB'] for s in S]):
    raise ValueError('Leaf bounds do not converge.')

# Cost evaluation
l_calc = 0
for n in range(N):
    l_calc += tasks[n].loss_fcn(t_ex_opt[n])
if abs(l_calc - l_opt) > 1e-12:
    raise ValueError('Iterated loss is inaccurate')

# Check solution validity
valid = True
for n_1 in range(N-1):
    for n_2 in range(n_1+1, N):
        cond_1 = t_ex_opt[n_1] >= (t_ex_opt[n_2] + tasks[n_2].duration)
        cond_2 = t_ex_opt[n_2] >= (t_ex_opt[n_1] + tasks[n_1].duration)
        valid = valid and (cond_1 or cond_2)
        if not valid:
            raise ValueError('Invalid Solution: Scheduling Conflict')

# Plots
t_plot = np.arange(0, np.ceil(t_ex_opt.max() + tasks[t_ex_opt.argmax()].duration), 0.01)
plt.figure(num='tasks', clear=True)
for n in range(N):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
    plt.gca().set(title='Task Losses', xlabel='t', ylabel='Loss')
    plt.grid(True)
    plt.legend()


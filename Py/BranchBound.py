"""
Branch and Bound simulation example
"""

import numpy as np
# import matplotlib.pyplot as plt

# from numpy import array, arange, empty
from numpy.random import rand

from functools import partial

from task_loss_fcns import loss_lin_drop

# %% Inputs

# Algorithm
mode_stack = 'LIFO'
# mode_stack = 'FIFO'

# Tasks
N = 10  # number of tasks

d_task = 1 + 2 * rand(N)

t_start = 30 * rand(N)
w = 0.8 + 0.4 * rand(N)
t_drop = t_start + d_task * (3 + 2 * rand(N))
l_drop = (2 + rand(N)) * w * (t_drop - t_start)

l_task = []
for n in range(N):
    l_task.append(partial(loss_lin_drop, w=w[n], t_start=t_start[n], t_drop=t_drop[n], l_drop=l_drop[n]))
    # l_task.append(lambda t: loss_lin_drop(t, w[n], t_start[n], t_drop[n], l_drop[n]))

# %% Tree Search

# TODO: add tic

# Initialize Stack
LB = 0
UB = 0
temp = sum(d_task) + max(t_start)
for n in range(N):
    LB += l_task[n](t_start[n])
    UB += l_task[n](temp)

# TODO: research numpy array use for named fields like 'struct'

# S0 = array([((3), tuple(np.repeat(np.inf, N)), 0, LB, UB)],
#            dtype=[('seq', 'int32'), ('t_ex', 'float64'), ('l_inc', 'float64'), ('LB', 'float64'), ('UB', 'float64')])

S = [{'seq': [], 't_ex': np.repeat(np.inf, N), 'l_inc': 0, 'LB': LB, 'UB': UB}]

# # Iterate
# while (len(S) != 1) or (len(S[0]['seq']) != N):
#     print(f'# Remaining Branches = {len(S)}')
#
#     # Extract Branch
#     for i in range(len(S)):
#         if len(S[i]['seq']) != N:
#             B = S.pop(i)
#             break
#
#     # Split Branch
#     T_c = 0

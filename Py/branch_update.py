"""
Function for generating new branches on a Branch-and-Bound tree search.
"""


def branch_update(branch, n, tasks):
    """Generate New Branch"""

    seq_prev = branch['seq']
    t_ex_prev = branch['t_ex']
    l_inc_prev = branch['l_inc']

    seq = seq_prev + [n]

    t_ex = t_ex_prev.copy()
    if len(seq_prev) == 0:
        t_ex[n] = tasks[n].t_start
    else:
        t_ex[n] = max(tasks[n].t_start, t_ex_prev[seq_prev[-1]] + tasks[seq_prev[-1]].duration)

    l_inc = l_inc_prev + tasks[n].loss_fcn(t_ex[n])

    N = len(tasks)
    T_c = list(set(range(N)) - set(seq))

    t_end = t_ex[n] + tasks[n].duration
    t_s_max = max([tasks[i].t_start for i in T_c] + [t_end]) + sum([tasks[i].duration for i in T_c])

    lb = l_inc
    ub = l_inc
    for n in T_c:
        lb += tasks[n].loss_fcn(max(tasks[n].t_start, t_end))
        ub += tasks[n].loss_fcn(t_s_max)

    return {'seq': seq, 't_ex': t_ex, 'l_inc': l_inc, 'LB': lb, 'UB': ub}
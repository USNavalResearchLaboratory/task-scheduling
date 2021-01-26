import numpy as np

import task_scheduling
from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms import base as algs_base
from task_scheduling import util


ch_avail = np.zeros(2, dtype=np.float)

# tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
tasks_full = [task_scheduling.tasks.ReluDrop(1, 0, slope, 5, 10) for slope in np.arange(1, 1.4, .1)]
# tasks_full = [task_scheduling.tasks.ReluDropRadar.search(0.018, 'AHS') for _ in range(4)]


# def get_tasks(tasks_):
#     tasks_sort = sorted(tasks_, key=lambda task_: task_.t_release)
#     return tasks_sort[:2]


def priority(task_):
    return -task_.t_release

# TODO: how to handle negative release times? Use shift node to reparameterize?

t_clock = 0.
# t_del = 0.01
loss_full = 0.
for __ in range(100):
    # tasks = get_tasks(tasks_full)
    tasks_full.sort(key=priority)
    tasks = tasks_full[-2:]

    t_ex, ch_ex = algs_base.earliest_release(tasks, ch_avail)
    # t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)

    # Scheduled task updates
    for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
        loss_full += task(t_ex_i)

        # TODO: drop, dont execute if t_ex is too high. Avoids unnecessary ch_avail increment

        task.t_release = t_ex_i + task.duration
        ch_avail[ch_ex_i] = max(ch_avail[ch_ex_i], task.t_release)      # TODO: get from TreeNode?

    # TODO: effectively jumps sim time to ch_avail_min

    # Dropped task updates
    ch_avail_min = min(ch_avail)
    for task in tasks_full:
        while task.t_release + task.t_drop < ch_avail_min:      # absolute drop time
            loss_full += task.l_drop        # add drop loss
            task.t_release += task.t_drop   # increment release time

    # t_clock += t_del

from functools import wraps
from operator import attrgetter

import numpy as np


def sort_wrapper(scheduler, sort_func):
    if isinstance(sort_func, str):
        sort_func = attrgetter(sort_func)

    @wraps(scheduler)
    def sorted_scheduler(tasks, ch_avail):
        idx = list(np.argsort([sort_func(task) for task in tasks]))
        idx_inv = [idx.index(n) for n in range(len(tasks))]

        sch = scheduler([tasks[i] for i in idx], ch_avail)
        return sch[idx_inv]

    return sorted_scheduler


# def timing_wrapper(scheduler):
#     """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""
#
#     @wraps(scheduler)
#     def timed_scheduler(tasks, ch_avail):
#         t_start = perf_counter()
#         # t_ex, ch_ex = scheduler(tasks, ch_avail)
#         # t_run = perf_counter() - t_start
#         # # return t_ex, ch_ex, t_run
#
#         solution = scheduler(tasks, ch_avail)
#         t_run = perf_counter() - t_start
#         return SchedulingSolution(*solution, t_run=t_run)
#
#     return timed_scheduler

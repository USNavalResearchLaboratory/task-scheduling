"""Algorithm wrappers."""

from functools import wraps
from operator import attrgetter

import numpy as np

from task_scheduling.util import evaluate_schedule


def sort_wrapper(scheduler, sort_func):
    """
    Create a chained scheduler with pre-sort function.

    Parameters
    ----------
    scheduler : callable
        Task scheduler.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.

    Returns
    -------
    callable

    """
    if isinstance(sort_func, str):
        sort_func = attrgetter(sort_func)

    @wraps(scheduler)
    def sorted_scheduler(tasks, ch_avail):
        idx = list(np.argsort([sort_func(task) for task in tasks]))
        idx_inv = [idx.index(n) for n in range(len(tasks))]

        sch = scheduler([tasks[i] for i in idx], ch_avail)
        return sch[idx_inv]

    return sorted_scheduler


def ensemble_scheduler(*schedulers):
    """Create function that evaluates multiple schedulers and returns the best solution."""

    def new_scheduler(tasks, ch_avail):
        sch_best = None
        loss_best = float("inf")
        for scheduler in schedulers:
            sch = scheduler(tasks, ch_avail)
            loss = evaluate_schedule(tasks, sch)
            if loss < loss_best:
                sch_best = sch
                loss_best = loss
        return sch_best

    return new_scheduler

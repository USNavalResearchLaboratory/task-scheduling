from task_scheduling.util import evaluate_schedule


def ensemble_scheduler(*schedulers):
    """Create function that evaluates multiple schedulers and returns the best solution."""

    def new_scheduler(tasks, ch_avail):
        t_ex_best, ch_ex_best = None, None
        l_ex_best = float('inf')
        for scheduler in schedulers:
            t_ex, ch_ex = scheduler(tasks, ch_avail)
            l_ex = evaluate_schedule(tasks, t_ex)
            if l_ex < l_ex_best:
                t_ex_best, ch_ex_best = t_ex, ch_ex
                l_ex_best = l_ex
        return t_ex_best, ch_ex_best

    return new_scheduler

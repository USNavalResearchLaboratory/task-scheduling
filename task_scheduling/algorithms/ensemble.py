from task_scheduling.util import evaluate_schedule


def ensemble_scheduler(*schedulers):
    """Create function that evaluates multiple schedulers and returns the best solution."""

    def new_scheduler(tasks, ch_avail):
        ex_best = None
        loss_best = float('inf')
        for scheduler in schedulers:
            ex = scheduler(tasks, ch_avail)
            loss = evaluate_schedule(tasks, ex)
            if loss < loss_best:
                ex_best = ex
                loss_best = loss
        return ex_best

    return new_scheduler

from functools import wraps
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_scheduling.base import SchedulingSolution


def summarize_tasks(tasks, **tabulate_kwargs):
    """Create and print a Pandas DataFrame detailing tasks."""
    cls_task = tasks[0].__class__
    if all(isinstance(task, cls_task) for task in tasks[1:]):
        df = pd.DataFrame([task.to_series() for task in tasks])
        tabulate_kwargs_ = {'tablefmt': 'github', 'floatfmt': '.3f'}
        tabulate_kwargs_.update(tabulate_kwargs)
        str_ = f"{cls_task.__name__}\n{df.to_markdown(**tabulate_kwargs_)}"
    else:
        str_ = '\n'.join(task.summary() for task in tasks)

    return str_


def plot_task_losses(tasks, t_plot=None, ax=None, ax_kwargs=None):
    """
    Plot loss functions for various tasks.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    t_plot : numpy.ndarray
        Loss evaluation times.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict, optional
        Additional Axes keyword parameters.

    """

    if t_plot is None:
        x_lim = min(task.plot_lim[0] for task in tasks), max(task.plot_lim[1] for task in tasks)
        t_plot = np.arange(*x_lim, 0.01)
    else:
        x_lim = t_plot[0], t_plot[-1]

    _temp = np.array([task(x_lim) for task in tasks])
    _temp[np.isnan(_temp)] = 0.
    y_lim = _temp[:, 0].min(), _temp[:, 1].max()

    if ax is None:
        _, ax = plt.subplots()
    else:
        x_lim_gca, y_lim_gca = ax.get_xlim(), ax.get_ylim()
        x_lim = min(x_lim[0], x_lim_gca[0]), max(x_lim[1], x_lim_gca[1])
        y_lim = min(y_lim[0], y_lim_gca[0]), max(y_lim[1], y_lim_gca[1])

    for task in tasks:
        task.plot_loss(t_plot, ax)

    if ax_kwargs is None:
        ax_kwargs = {}
    ax_kwargs = dict(xlabel='t', ylabel='Loss', xlim=x_lim, ylim=y_lim) | ax_kwargs
    ax.set(**ax_kwargs)

    ax.legend()


def check_schedule(tasks, sch, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    sch : numpy.ndarray
        Task execution schedule.
    tol : float, optional
        Time tolerance for validity conditions.

    Raises
    -------
    ValueError
        If tasks overlap in time.

    """

    for c in np.unique(sch['c']):
        tasks_ch = np.array(tasks)[sch['c'] == c].tolist()
        t_ch = sch['t'][sch['c'] == c]
        for n_1 in range(len(tasks_ch)):
            if t_ch[n_1] + tol < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                conditions = [t_ch[n_1] + tol < t_ch[n_2] + tasks_ch[n_2].duration,
                              t_ch[n_2] + tol < t_ch[n_1] + tasks_ch[n_1].duration]
                if all(conditions):
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def evaluate_schedule(tasks, sch):
    """
    Evaluate scheduling loss.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks
    sch : Collection of float
        Task execution schedule.

    Returns
    -------
    float
        Total loss of scheduled tasks.

    """

    loss = 0.
    for task, t in zip(tasks, sch['t']):
        loss += task(t)

    return loss


def plot_schedule(tasks, sch, n_ch=None, loss=None, name=None, ax=None, ax_kwargs=None):
    """
    Plot task schedule.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    sch : numpy.ndarray
        Task execution schedule.
    n_ch : int, optional
        Number of channels.
    loss : float or None
        Total loss of scheduled tasks.
    name : str or None
        Algorithm string representation
    ax : Axes or None
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """
    if ax is None:
        _, ax = plt.subplots()

    cycle = plt.rcParams['axes.prop_cycle']()
    for n, task in enumerate(tasks):
        ax.broken_barh([(sch['t'][n], task.duration)], (sch['c'][n] - 0.5, 1),
                       facecolors=next(cycle)['color'], edgecolor='black', label=task)

    # x_lim = min(sch['t']), max(task.duration + t for task, t in zip(tasks, sch['t']))
    if np.isnan(sch['t']).all():
        x_lim = (0, plt.rcParams['axes.xmargin'])
    else:
        x_lim = np.nanmin(sch['t']), np.nanmax([task.duration + t for task, t in zip(tasks, sch['t'])])

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if loss is not None:
        _temp.append(f'Loss = {loss:.3f}')
    title = ', '.join(_temp)

    if n_ch is None:  # infer from `sch`
        n_ch = sch['c'].max() + 1

    if ax_kwargs is None:
        ax_kwargs = {}
    ax_kwargs = dict(xlim=x_lim, ylim=(-.5, n_ch - 1 + .5), xlabel='t', yticks=list(range(n_ch)), ylabel='Channel',
                     title=title) | ax_kwargs
    ax.set(**ax_kwargs)

    ax.legend()


def eval_wrapper(scheduler):
    """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        sch = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        check_schedule(tasks, sch)
        loss = evaluate_schedule(tasks, sch)

        return SchedulingSolution(sch, loss, t_run)

    return timed_scheduler

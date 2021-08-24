from functools import wraps
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_scheduling._core import SchedulingSolution


def check_schedule(tasks, t_ex, ch_ex, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : numpy.ndarray
        Task execution times.
    ch_ex : numpy.ndarray
        Task execution channels.
    tol : float, optional
        Time tolerance for validity conditions.

    Raises
    -------
    ValueError
        If tasks overlap in time.

    """

    # if np.isnan(t_ex).any():
    #     raise ValueError("All tasks must be scheduled.")

    for ch in np.unique(ch_ex):
        tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch)):
            if t_ex_ch[n_1] + tol < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                conditions = [t_ex_ch[n_1] + tol < t_ex_ch[n_2] + tasks_ch[n_2].duration,
                              t_ex_ch[n_2] + tol < t_ex_ch[n_1] + tasks_ch[n_1].duration]
                if all(conditions):
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def evaluate_schedule(tasks, t_ex):
    """
    Evaluate scheduling loss.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
        Tasks
    t_ex : Sequence of float
        Task execution times.

    Returns
    -------
    float
        Total loss of scheduled tasks.

    """

    l_ex = 0.
    for task, t_ex in zip(tasks, t_ex):
        l_ex += task(t_ex)

    return l_ex


def eval_wrapper(scheduler):
    """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        t_ex, ch_ex, *__ = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        check_schedule(tasks, t_ex, ch_ex)
        l_ex = evaluate_schedule(tasks, t_ex)

        return SchedulingSolution(t_ex, ch_ex, l_ex, t_run)

    return timed_scheduler


def summarize_tasks(tasks, file=None, **tabulate_kwargs):
    """Create and print a Pandas DataFrame detailing tasks."""
    df = pd.DataFrame([task.to_series() for task in tasks])
    tabulate_kwargs_ = {'tablefmt': 'github', 'floatfmt': '.3f'}
    tabulate_kwargs_.update(tabulate_kwargs)
    print(df.to_markdown(**tabulate_kwargs_), file=file)


def plot_task_losses(tasks, t_plot=None, ax=None, ax_kwargs=None):
    """
    Plot loss functions for a list of tasks.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_plot : numpy.ndarray
        Loss evaluation times.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict, optional
        Additional Axes keyword parameters.

    """
    if ax_kwargs is None:
        ax_kwargs = {}

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

    ax.set(xlabel='t', ylabel='Loss')
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.legend()
    ax.set(**ax_kwargs)


def plot_schedule(tasks, t_ex, ch_ex, l_ex=None, name=None, ax=None, ax_kwargs=None):
    """
    Plot task schedule.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : numpy.ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : numpy.ndarray
        Task execution channels. NaN for unscheduled.
    l_ex : float or None
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

    if ax_kwargs is None:
        ax_kwargs = {}

    n_ch = len(np.unique(ch_ex))
    bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    for n, task in enumerate(tasks):
        label = str(task)
        # label = f'Task #{n}'
        ax.broken_barh([(t_ex[n], task.duration)], (ch_ex[n] - 0.5, 1),
                       facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=label)

    x_lim = min(t_ex), max(t_ex[n] + task.duration for n, task in enumerate(tasks))
    ax.set(xlim=x_lim, ylim=(-.5, n_ch - 1 + .5), xlabel='t',
           yticks=list(range(n_ch)), ylabel='Channel')

    ax.legend()

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if l_ex is not None:
        _temp.append(f'Loss = {l_ex:.3f}')
    title = ', '.join(_temp)
    if len(title) > 0:
        ax.set_title(title)

    ax.set(**ax_kwargs)

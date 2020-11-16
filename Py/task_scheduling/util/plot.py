import numpy as np
import matplotlib.pyplot as plt


def plot_task_losses(tasks, t_plot=None, ax=None, ax_kwargs=None):
    """
    Plot loss functions for a list of tasks.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Generic
    t_plot : ndarray
        Loss evaluation times.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict, optional
        Additional Axes keyword parameters.

    Returns
    -------

    """
    if ax_kwargs is None:
        ax_kwargs = {}

    if t_plot is None:
        x_lim = min(task.plot_lim[0] for task in tasks), max(task.plot_lim[1] for task in tasks)
        t_plot = np.arange(*x_lim, 0.01)

    x_lim = t_plot[0], t_plot[-1]
    y_lim = min(task(x_lim[0]) for task in tasks), max(task(x_lim[1]) for task in tasks)

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
    ax.grid(True)
    ax.legend()
    ax.set(**ax_kwargs)


def plot_schedule(tasks, t_ex, ch_ex, l_ex=None, name=None, ax=None, ax_kwargs=None):
    """
    Plot task schedule.

    Parameters
    ----------
    tasks : list of Generic
    t_ex : ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : ndarray
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
        ax.broken_barh([(t_ex[n], task.duration)], (ch_ex[n] - 0.5, 1),
                       facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=f'Task #{n}')

    x_lim = min(t_ex), max(t_ex[n] + task.duration for n, task in enumerate(tasks))
    ax.set(xlim=x_lim, ylim=(-.5, n_ch - 1 + .5), xlabel='t',
           yticks=list(range(n_ch)), ylabel='Channel')
    ax.grid(True)
    # ax.legend()

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if l_ex is not None:
        _temp.append(f'Loss = {l_ex:.2f}')
    title = ', '.join(_temp)
    if len(title) > 0:
        ax.set_title(title)

    ax.set(**ax_kwargs)


def scatter_loss_runtime(t_run, l_ex, ax=None, ax_kwargs=None):
    """
    Scatter plot of total execution loss versus runtime.

    Parameters
    ----------
    t_run : ndarray
        Runtime of algorithm.
    l_ex : ndarray
        Total loss of scheduled tasks.
    ax : Axes or None
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """
    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    for name in t_run.dtype.names:
        ax.scatter(t_run[name], l_ex[name], label=name)

    ax.set(xlabel='Runtime (s)', ylabel='Loss')
    ax.grid(True)
    ax.legend()
    ax.set(**ax_kwargs)


def plot_loss_runtime(t_run, l_ex, do_std=False, ax=None, ax_kwargs=None):
    """
    Line plot of total execution loss versus maximum runtime.

    Parameters
    ----------
    t_run : ndarray
        Runtime of algorithm.
    l_ex : ndarray
        Total loss of scheduled tasks.
    do_std : bool
        Activates error bars for sample standard deviation.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """

    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    names = l_ex.dtype.names
    for i_name, name in enumerate(names):
        l_mean = l_ex[name].mean(-1)
        ax.plot(t_run, l_mean, label=name)
        if do_std:
            l_std = l_ex[name].std(-1)
            # ax.errorbar(t_run, l_mean, yerr=l_std, label=name, errorevery=(i_name, len(names)))
            ax.fill_between(t_run, l_mean - l_std, l_mean + l_std, alpha=0.25)
        # else:
        #     ax.plot(t_run, l_mean, label=name)

    ax.set(xlabel='Runtime (s)', ylabel='Loss')
    ax.grid(True)
    ax.legend()
    ax.set(**ax_kwargs)

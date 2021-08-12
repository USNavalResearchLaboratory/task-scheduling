import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

import numpy as np
import matplotlib.pyplot as plt

from util.results import eval_loss


def plot_task_losses(tasks, t_plot=None, ax=None):
    """Plot loss functions for a list of tasks."""

    if t_plot is None:
        x_lim = min([task.plot_lim[0] for task in tasks]), max([task.plot_lim[1] for task in tasks])
        t_plot = np.arange(*x_lim, 0.01)

    x_lim = t_plot[0], t_plot[-1]
    y_lim = min([task.loss_fcn(x_lim[0]) for task in tasks]), max([task.loss_fcn(x_lim[1]) for task in tasks])
    # y_lim = 0, 1 + max([task.loss_fcn(float('inf')) for task in tasks])

    if ax is None:
        _, ax = plt.subplots()
    else:
        x_lim_gca, y_lim_gca = ax.get_xlim(), ax.get_ylim()
        x_lim = min([x_lim[0], x_lim_gca[0]]), max([x_lim[1], x_lim_gca[1]])
        y_lim = min([y_lim[0], y_lim_gca[0]]), max([y_lim[1], y_lim_gca[1]])

    for task in tasks:
        task.plot_loss(t_plot, ax)

    ax.set(xlabel='t', ylabel='Loss')
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid(True)
    ax.legend()


def plot_schedule(tasks, t_ex, ch_ex, l_ex=None, alg_str=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    n_ch = len(np.unique(ch_ex))
    bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    for n, task in enumerate(tasks):
        ax.broken_barh([(t_ex[n], task.duration)], (ch_ex[n] - 0.5, 1),
                       facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=f'Task #{n}')

    x_lim = min(t_ex), max([t_ex[n] + task.duration for n, task in enumerate(tasks)])
    ax.set(xlim=x_lim, ylim=(-.5, n_ch - 1 + .5), xlabel='t',
           yticks=list(range(n_ch)), ylabel='Channel')
    ax.grid(True)
    # ax.legend()

    if l_ex is None:
        l_ex = eval_loss(tasks, t_ex)

    if alg_str is None:
        ax.set_title(f'Loss = {l_ex:.3f}')
    else:
        ax.set_title(f'{alg_str}: Loss = {l_ex:.3f}')


def plot_results(alg_str, t_run, l_ex, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    for i, alg in enumerate(alg_str):
        ax.scatter(t_run[i], l_ex[i], label=alg_str[i])

    ax.set(xlabel='Runtime (s)', ylabel='Loss')
    ax.grid(True)
    ax.legend()

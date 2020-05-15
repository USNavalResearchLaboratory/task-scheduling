import numpy as np
import matplotlib.pyplot as plt


def check_rng(rng):
    """Return a random number generator."""
    if rng is None:
        return np.random.default_rng()
    elif type(rng) == int:
        return np.random.default_rng(rng)
    else:
        return rng      # TODO: type check? assumes valid rng


def check_valid(tasks, t_ex, ch_ex):
    """Check schedule validity."""

    for ch in np.unique(ch_ex):
        tasks_ch = np.asarray(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch) - 1):
            for n_2 in range(n_1 + 1, len(tasks_ch)):
                if t_ex_ch[n_1] - tasks_ch[n_2].duration + 1e-12 < t_ex_ch[n_2] < t_ex_ch[n_1] \
                        + tasks_ch[n_1].duration - 1e-12:
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def eval_loss(tasks, t_ex):
    """Evaluate scheduling loss."""

    l_ex = 0
    # for n in range(len(tasks)):
    #     l_ex += tasks[n].loss_fcn(t_ex[n])
    for task, t_ex in zip(tasks, t_ex):
        l_ex += task.loss_fcn(t_ex)

    return l_ex


def plot_task_losses(tasks, t_plot=None, ax=None):
    if t_plot is None:
        x_lim_max = max([task._plot_lim[-1] for task in tasks])
        t_plot = np.arange(0, x_lim_max, 0.01)

    if ax is None:
        _, ax = plt.subplots()
        ax.set(xlabel='t', ylabel='Loss')
        y_lim_max = 1 + max([task.loss_fcn(float('inf')) for task in tasks])
        ax.set_ylim(0, y_lim_max)
        ax.set_xlim(t_plot[[0, -1]])
        ax.grid(True)

    for task in tasks:
        task.plot_loss(t_plot, ax)

    ax.legend()


def plot_schedules():
    pass

    # bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # for i in range(len(algorithms)):
    #     title_dict = algorithms[i].keywords
    #     for key in ['verbose', 'rng', 'ch_avail']:
    #         try:
    #             del title_dict[key]
    #         except KeyError:
    #             pass
    #     title = ": ".join([algorithms[i].func.__name__, str(title_dict)])
    #
    #     plt.figure(num=title, clear=True, figsize=[8, 2.5])
    #     plt.title(f'Loss = {l_ex_alg[i]:.3f}')
    #     # d = ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    #     for n in range(len(tasks)):
    #         plt.gca().broken_barh([(t_ex_alg[i][n], tasks[n].duration)], (ch_ex_alg[i][n]-0.5, 1),
    #                               facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=f'Task #{n}')
    #
    #     plt.gca().set(xlim=t_plot[[0, -1]], ylim=(-.5, n_channels-1+.5),
    #                   xlabel='t', yticks=list(range(n_channels)), ylabel='Channel')
    #     plt.gca().grid(True)
    #     plt.gca().legend()


import numpy as np
import matplotlib.pyplot as plt


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
        l_ex += tasks.loss_fcn(t_ex)

    return l_ex



# %% Graphics
t_plot_max = 0
for t_ex in t_ex_alg:
    t_plot_max = max(t_plot_max, max(t_ex))
t_plot_max += max([t.duration for t in tasks])

t_plot = np.arange(0, t_plot_max, 0.01)

plt.figure(num='Task Loss Functions', clear=True)
for n in range(n_tasks):
    plt.plot(t_plot, tasks[n].loss_fcn(t_plot), label=f'Task #{n}')
plt.gca().set(xlabel='t', ylabel='Loss')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(t_plot[[0, -1]])
plt.grid(True)
plt.legend()


bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(algorithms)):
    title_dict = algorithms[i].keywords
    for key in ['verbose', 'rng', 'ch_avail']:
        try:
            del title_dict[key]
        except KeyError:
            pass
    title = ": ".join([algorithms[i].func.__name__, str(title_dict)])

    plt.figure(num=title, clear=True, figsize=[8, 2.5])
    plt.title(f'Loss = {l_ex_alg[i]:.3f}')
    # d = ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    for n in range(len(tasks)):
        plt.gca().broken_barh([(t_ex_alg[i][n], tasks[n].duration)], (ch_ex_alg[i][n]-0.5, 1),
                              facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=f'Task #{n}')

    plt.gca().set(xlim=t_plot[[0, -1]], ylim=(-.5, n_channels-1+.5),
                  xlabel='t', yticks=list(range(n_channels)), ylabel='Channel')
    plt.gca().grid(True)
    plt.gca().legend()
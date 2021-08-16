def scatter_loss_runtime_stats(t_run, l_ex, ax=None, ax_kwargs=None):
    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    for name in t_run.dtype.names:
        color = next(ax._get_lines.prop_cycler)['color']

        # ax.scatter(t_run[name], l_ex[name], label=name)

        x_mean = np.mean(t_run[name])
        y_mean = np.mean(l_ex[name])
        ax.scatter(x_mean, y_mean, label=name + '_mean', color=color, marker='*', s=400)

        # x_median = np.median(t_run[name])
        # y_median = np.median(l_ex[name])
        # ax.scatter(x_median, y_median, label=name + '_median', color=color, marker='d', s=400)

        # x_std = np.std(t_run[name])
        # y_std = np.std(l_ex[name])
        # ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, color=color, capsize=2)

    ax.set(xlabel='Runtime (s)', ylabel='Loss')
    ax.legend()
    ax.set(**ax_kwargs)
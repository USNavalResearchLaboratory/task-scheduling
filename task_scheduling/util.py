"""Common utilities."""

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
        # TODO: do grouping even if not all are same type
        df = pd.DataFrame([task.to_series() for task in tasks])
        tabulate_kwargs_ = dict(tablefmt="github", floatfmt=".3f")
        tabulate_kwargs_.update(tabulate_kwargs)
        str_ = f"{cls_task.__name__}\n{df.to_markdown(**tabulate_kwargs_)}"
    else:
        str_ = "\n".join(task.summary() for task in tasks)

    return str_


def plot_task_losses(tasks, t_plot=None, ax=None, ax_kwargs=None, legend=False):
    """
    Plot loss functions for various tasks.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks.
    t_plot : numpy.ndarray
        Loss evaluation times.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict, optional
        Additional Axes keyword parameters.
    legend : bool, optional
        Add legend to plot.

    """
    if t_plot is None:
        lows, highs = zip(*(task.plot_lim for task in tasks))
        t_plot = np.arange(min(lows), max(highs), 1e-3)

    with plt.rc_context({"axes.xmargin": 0}):
        if ax is None:
            _, ax = plt.subplots()
        for task in tasks:
            task.plot_loss(t_plot, ax)

    if ax_kwargs is None:
        ax_kwargs = {}
    ax_kwargs = dict(xlabel="$t$", ylabel="$l$") | ax_kwargs
    ax.set(**ax_kwargs)
    ax.set_ylim(bottom=0.0)

    if legend:
        ax.legend()


def check_schedule(tasks, sch, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks.
    sch : numpy.ndarray
        Task execution schedule.
    tol : float, optional
        Time tolerance for validity conditions.

    Raises
    ------
    ValueError
        If tasks overlap in time.

    """
    for c in np.unique(sch["c"]):
        tasks_ch = np.array(tasks)[sch["c"] == c].tolist()
        t_ch = sch["t"][sch["c"] == c]
        for n_1 in range(len(tasks_ch)):
            if t_ch[n_1] + tol < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                conditions = [
                    t_ch[n_1] + tol < t_ch[n_2] + tasks_ch[n_2].duration,
                    t_ch[n_2] + tol < t_ch[n_1] + tasks_ch[n_1].duration,
                ]
                if all(conditions):
                    raise ValueError("Invalid Solution: Scheduling Conflict")


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
    return sum(task(t) for task, t in zip(tasks, sch["t"]))


def plot_schedule(
    tasks, sch, n_ch=None, loss=None, name=None, ax=None, ax_kwargs=None, legend=False
):
    """
    Plot task schedule.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks.
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
    legend : bool, optional
        Add legend to plot.

    """
    if ax is None:
        _, ax = plt.subplots()

    cycle = plt.rcParams["axes.prop_cycle"]()
    for n, task in enumerate(tasks):
        ax.broken_barh(
            [(sch["t"][n], task.duration)],
            (sch["c"][n] - 0.5, 1),
            facecolors=next(cycle)["color"],
            edgecolor="black",
            label=str(task),
        )

    if np.isnan(sch["t"]).all():
        x_lim = (0, plt.rcParams["axes.xmargin"])
    else:
        x_lim = (
            np.nanmin(sch["t"]),
            np.nanmax([task.duration + t for task, t in zip(tasks, sch["t"])]),
        )

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if loss is not None:
        _temp.append(f"$L = {loss:.3f}$")
    title = ", ".join(_temp)

    if n_ch is None:
        n_ch = sch["c"].max() + 1  # infer from `sch`

    ax_kwargs_base = dict(
        xlim=x_lim,
        ylim=(-0.5, n_ch - 1 + 0.5),
        xlabel="$t$",
        yticks=list(range(n_ch)),
        ylabel="$c$",
        title=title,
    )
    if ax_kwargs is None:
        ax_kwargs = {}
    ax_kwargs = ax_kwargs_base | ax_kwargs
    ax.set(**ax_kwargs)

    if legend:
        ax.legend()


def eval_wrapper(scheduler):
    """Wrap a scheduler, creating a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        sch = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        check_schedule(tasks, sch)
        loss = evaluate_schedule(tasks, sch)

        return SchedulingSolution(sch, loss, t_run)

    return timed_scheduler


def plot_losses_and_schedule(
    tasks, sch, n_ch=None, loss=None, name=None, fig_kwargs=None, legend=False
):
    """
    Plot task loss functions with schedule, including partial losses.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
        Tasks.
    sch : numpy.ndarray
        Task execution schedule.
    n_ch : int, optional
        Number of channels.
    loss : float or None
        Total loss of scheduled tasks.
    name : str or None
        Algorithm string representation
    fig_kwargs : dict, optional
        `matplotlib.Figure` arguments.
    legend : bool, optional
        Add legend to plot.

    """
    if fig_kwargs is None:
        fig_kwargs = {}
        # fig_kwargs = dict(gridspec_kw=dict(left=0.1, right=0.85))

    with plt.rc_context({"axes.xmargin": 0}):
        fig, axes = plt.subplots(2, clear=True, **fig_kwargs)

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if loss is not None:
        _temp.append(f"$L = {loss:.3f}$")
    fig.suptitle(", ".join(_temp), y=0.95)

    plot_schedule(
        tasks, sch, n_ch, loss, name=None, ax=axes[1], ax_kwargs=dict(title=""), legend=legend
    )

    lows, highs = zip(axes[1].get_xlim(), *(task.plot_lim for task in tasks))
    t_plot = np.arange(min(lows), max(highs), 1e-3)
    plot_task_losses(tasks, t_plot, ax=axes[0], ax_kwargs=dict(xlabel=""), legend=legend)

    # Mark loss functions with execution times
    for task, (t_ex, _), line in zip(tasks, sch, axes[0].get_lines()):
        axes[0].plot(
            [t_ex], [task(t_ex)], color=line.get_color(), marker="o", linestyle="", label=None
        )

    # Match x-axis limits
    lows, highs = zip(*(ax.get_xlim() for ax in axes))
    x_lims = min(lows), max(highs)
    for ax in axes:
        ax.set(xlim=x_lims)

    if legend:
        # Use single `Figure` legend
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )
        for ax in axes:
            ax.get_legend().remove()

    return fig

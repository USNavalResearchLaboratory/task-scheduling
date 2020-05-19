"""Task objects."""

# TODO: document class attributes, even if identical to init parameters?

import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_rng


class GenericTask:
    """
    Generic task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.

    """

    def __init__(self, duration, t_release, loss_fcn):
        self.duration = duration
        self.t_release = t_release
        self.loss_fcn = loss_fcn

        self.plot_lim = (t_release, t_release + duration)

    def __repr__(self):
        return f"GenericTask(duration: {self.duration:.2f}, release time:{self.t_release:.2f})"

    def plot_loss(self, t_plot=None, ax=None):
        if t_plot is None:
            t_plot = np.arange(*self.plot_lim, 0.01)
        elif t_plot[0] < self.t_release:
            t_plot = t_plot[t_plot >= self.t_release]

        x_lim = t_plot[0], t_plot[-1]
        y_lim = self.loss_fcn(list(x_lim))
        # y_lim = 0, 1 + self.loss_fcn(float('inf'))

        if ax is None:
            _, ax = plt.subplots()

            ax.set(xlabel='t', ylabel='Loss')
            plt.grid(True)
            plt.title(self.__repr__())
        else:
            x_lim_gca, y_lim_gca = ax.get_xlim(), ax.get_ylim()
            x_lim = min([x_lim[0], x_lim_gca[0]]), max([x_lim[1], x_lim_gca[1]])
            y_lim = min([y_lim[0], y_lim_gca[0]]), max([y_lim[1], y_lim_gca[1]])

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        plot_data = ax.plot(t_plot, self.loss_fcn(t_plot), label=self.__repr__())

        return plot_data


class ReluDropTask(GenericTask):
    """
    Generates a rectified linear loss function with a constant drop penalty.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled. Loss at t_release is zero.
    slope : float
        Function slope between release and drop times.
    t_drop : float
        Drop time.
    l_drop : float
        Constant loss after drop time.

    """

    def __init__(self, duration, t_release, slope, t_drop, l_drop):
        super().__init__(duration, t_release, loss_relu_drop(t_release, slope, t_drop, l_drop))
        self.slope = slope
        self.t_drop = t_drop
        self.l_drop = l_drop

        self.plot_lim = (t_release, t_drop + duration)

        if l_drop < slope * (t_drop - t_release):
            raise ValueError("Function is not monotonically non-decreasing.")

    def __repr__(self):
        return f"ReluDropTask(duration: {self.duration:.2f}, release time:{self.t_release:.2f})"


def loss_relu_drop(t_release, slope, t_drop, l_drop):
    """
    Rectified linear loss function with a constant drop penalty.

    Parameters
    ----------
    t_release : float
        Earliest time the task may be scheduled. Loss at t_release is zero.
    slope : float
        Function slope between release and drop times.
    t_drop : float
        Drop time.
    l_drop : float
        Constant loss after drop time.

    Returns
    -------
    function
        Incurred loss

    """

    def loss_fcn(t):
        t = np.asarray(t)[np.newaxis]
        loss = slope * (t - t_release)
        loss[t < t_release] = np.inf
        loss[t >= t_drop] = l_drop
        return loss.squeeze(axis=0)

    return loss_fcn


# %% Task generation objects        # TODO: generalize, docstrings
class GenericTaskGenerator:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def rand_tasks(self, n_tasks):
        raise NotImplementedError


class ReluDropGenerator(GenericTaskGenerator):
    def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        super().__init__(rng)
        self.duration_lim = duration_lim
        self.t_release_lim = t_release_lim
        self.slope_lim = slope_lim
        self.t_drop_lim = t_drop_lim
        self.l_drop_lim = l_drop_lim

    def rand_tasks(self, n_tasks):
        duration = self.rng.uniform(*self.duration_lim, n_tasks)
        t_release = self.rng.uniform(*self.t_release_lim, n_tasks)
        slope = self.rng.uniform(*self.slope_lim, n_tasks)
        t_drop = self.rng.uniform(*self.t_drop_lim, n_tasks)
        l_drop = self.rng.uniform(*self.l_drop_lim, n_tasks)

        return [ReluDropTask(*args) for args in zip(duration, t_release, slope, t_drop, l_drop)]

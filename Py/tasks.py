"""Task objects."""

# TODO: document class attributes, even if identical to init parameters?

import numpy as np
import matplotlib.pyplot as plt
from util.utils import check_rng


class BaseTask:
    """Generic task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.

    """

    def __init__(self, duration, t_release):
        self.duration = duration
        self.t_release = t_release

        self._plot_lim = (0, 1)

    def __repr__(self):
        return f"BaseTask(duration: {self.duration:.3f}, release time:{self.t_release:.3f})"

    def loss_fcn(self, t):
        raise NotImplementedError   # TODO: add function to init for generic task creation?

    def plot_loss(self, t_plot=None, ax=None):
        if t_plot is None:
            t_plot = np.arange(*self._plot_lim, 0.01)

        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel='t', ylabel='Loss')
            ax.set_ylim(0, 1 + self.loss_fcn(float('inf')))
            ax.set_xlim(t_plot[[0, -1]])
            plt.grid(True)
            plt.title(self.__repr__())

        plot_data = ax.plot(t_plot, self.loss_fcn(t_plot), label=self.__repr__())

        return plot_data


class ReluDropTask(BaseTask):
    """Generates a rectified linear loss function with a constant drop penalty.

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
        super().__init__(duration, t_release)
        self.slope = slope
        self.t_drop = t_drop
        self.l_drop = l_drop

        self._plot_lim = (0, t_drop + duration)

        if l_drop < slope * (t_drop - t_release):
            raise ValueError("Function is not monotonically non-decreasing.")

    def __repr__(self):
        return f"ReluDropTask(duration: {self.duration:.3f}, release time:{self.t_release:.3f})"

    def loss_fcn(self, t):
        """Rectified linear loss function with a constant drop penalty.

        Parameters
        ----------
        t : ndarray
            Evaluation time

        Returns
        -------
        ndarray
            Incurred loss

        """

        t = np.asarray(t)[np.newaxis]
        loss = self.slope * (t - self.t_release)
        loss[t < self.t_release] = np.inf
        loss[t >= self.t_drop] = self.l_drop

        return loss.squeeze(axis=0)




# %% Task generation objects        # TODO: generalize, docstrings

class BaseTaskGenerator:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def rand_tasks(self, n_tasks, return_params=False):
        raise NotImplementedError


class ReluDropGenerator(BaseTaskGenerator):
    def __init__(self, rng=None):
        super().__init__(rng)

    def rand_tasks(self, n_tasks, return_params=False):
        duration = self.rng.uniform(1, 3, n_tasks)

        t_release = self.rng.uniform(0, 8, n_tasks)

        slope = self.rng.uniform(0.8, 1.2, n_tasks)
        t_drop = t_release + duration * self.rng.uniform(3, 5, n_tasks)
        l_drop = self.rng.uniform(2, 3, n_tasks) * slope * (t_drop - t_release)

        _params = list(zip(duration, t_release, slope, t_drop, l_drop))
        params = np.array(_params, dtype=[('duration', np.float), ('t_release', np.float),
                                          ('slope', np.float), ('t_drop', np.float), ('l_drop', np.float)])

        # tasks = [BaseTask.relu_drop(*args) for args in _params]
        tasks = [ReluDropTask(*args) for args in _params]

        if return_params:
            return tasks, params
        else:
            return tasks

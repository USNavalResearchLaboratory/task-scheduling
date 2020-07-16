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
    t_release : float
        Earliest time the task may be scheduled.
    duration : float
        Time duration of the task.
    loss_func : function, optional
        Maps execution time to scheduling loss.

    """

    def __init__(self, t_release, duration, loss_func=None):
        self.t_release = t_release
        self.duration = duration

        self._loss_func = loss_func

    def __repr__(self):
        return f"GenericTask(release time:{self.t_release:.2f}, duration: {self.duration:.2f})"

    def __call__(self, t):
        """Loss function versus time."""
        return self._loss_func(t)

    def shift_origin(self, t):
        raise NotImplementedError

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + self.duration

    def plot_loss(self, t_plot=None, ax=None):
        """
        Plot loss function.

        Parameters
        ----------
        t_plot : ndarray, optional
            Series of times for loss evaluation.
        ax : matplotlib.axes.Axes, optional
            Plotting axes object.
        """

        if t_plot is None:
            t_plot = np.arange(*self.plot_lim, 0.01)
        elif t_plot[0] < self.t_release:
            t_plot = t_plot[t_plot >= self.t_release]

        x_lim = t_plot[0], t_plot[-1]
        y_lim = self(list(x_lim))
        # y_lim = 0, 1 + self(float('inf'))

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

        plot_data = ax.plot(t_plot, self(t_plot), label=self.__repr__())

        return plot_data


class ReluDropTask(GenericTask):
    """
    Generates a rectified linear loss function task with a constant drop penalty.

    Parameters
    ----------
    t_release : float
        Earliest time the task may be scheduled.
    duration : float
        Time duration of the task.
    slope : float
        Function slope between release and drop times. Loss at release time is zero.
    t_drop : float
        Drop time relative to release time.
    l_drop : float
        Constant loss after drop time.

    """

    def __init__(self, t_release, duration, slope, t_drop, l_drop):
        super().__init__(t_release, duration)
        self._slope = slope
        self._t_drop = t_drop
        self._l_drop = l_drop

    def __repr__(self):
        return f"ReluDropTask(release time:{self.t_release:.2f}, duration: {self.duration:.2f})"

    def __call__(self, t):
        """Loss function versus time."""
        t = np.asarray(t)[np.newaxis] - self.t_release      # relative time

        loss = self.slope * t
        loss[t < 0] = np.inf
        loss[t >= self.t_drop] = self.l_drop
        return loss.squeeze(axis=0)

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, slope):
        # if slope <= 0:
        #     raise ValueError("Slope must be positive.")       # FIXME
        self.check_non_decreasing(slope, self.t_drop, self.l_drop)
        self._slope = slope

    @property
    def t_drop(self):
        return self._t_drop

    @t_drop.setter
    def t_drop(self, t_drop):
        self.check_non_decreasing(self.slope, t_drop, self.l_drop)
        self._t_drop = t_drop

    @property
    def l_drop(self):
        return self._l_drop

    @l_drop.setter
    def l_drop(self, l_drop):
        self.check_non_decreasing(self.slope, self.t_drop, l_drop)
        self._l_drop = l_drop

    @staticmethod
    def check_non_decreasing(slope, t_drop, l_drop):
        if l_drop < slope * t_drop:
            raise ValueError("Loss function must be monotonically non-decreasing.")

    def shift_origin(self, t):
        """Re-parameterize task after time elapses. Return loss incurred."""

        if t <= 0:
            raise ValueError("Shift time must be positive.")

        if self.is_dropped:
            return 0.

        t_excess = t - self.t_release
        self.t_release = max(0., -t_excess)
        if self.is_available:
            # Loss is incurred, drop time and loss are updated
            loss_inc = self(t_excess)
            self.t_drop = max(0., self.t_drop - t_excess)
            self.l_drop = self.l_drop - loss_inc
            return loss_inc
        else:
            return 0.   # No loss incurred

    @property
    def is_available(self):     # TODO: uses?? array rep?
        return self.t_release == 0.

    @property
    def is_dropped(self):
        return self.is_available and self.t_drop == 0.

    @property
    def param_rep(self):
        """An array representation of the parametric task."""

        # _params = [(task.duration, task.t_release, task.slope, task.t_drop, task.l_drop) for task in tasks]
        # params = np.array(_params, dtype=[('duration', np.float), ('t_release', np.float),
        #                                   ('slope', np.float), ('t_drop', np.float), ('l_drop', np.float)])
        # params.view(np.float).reshape(*params.shape, -1)
        return self.t_release, self.duration, self.slope, self.t_drop, self.l_drop    # TODO: add dropped flag?

    def summary(self):
        print(f'ReluDropTask\n------------\nrelease time: {self.t_release:.2f}'
              f'\nduration: {self.duration:.2f}\nslope: {self.slope:.2f}'
              f'\ndrop time: {self.t_drop:.2f}\ndrop loss: {self.l_drop:.2f}')

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + max(self.duration, self.t_drop)


# def loss_relu_drop(t_release, slope, t_drop, l_drop):
#     """
#     Rectified linear loss function with a constant drop penalty.
#
#     Parameters
#     ----------
#     t_release : float
#         Earliest time the task may be scheduled. Loss at t_release is zero.
#     slope : float
#         Function slope between release and drop times.
#     t_drop : float
#         Drop time.
#     l_drop : float
#         Constant loss after drop time.
#
#     Returns
#     -------
#     function
#         Incurred loss
#
#     """
#
#     def loss_func(t):
#         t = np.asarray(t)[np.newaxis]
#         loss = slope * (t - t_release)
#         loss[t < t_release] = np.inf
#         loss[t >= t_drop] = l_drop
#         return loss.squeeze(axis=0)
#
#     return loss_func


# %% Task generation objects        # TODO: generalize, add docstrings
class GenericTaskGenerator:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_tasks):
        raise NotImplementedError


class ReluDropGenerator(GenericTaskGenerator):
    def __init__(self, t_release_lim, duration_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        super().__init__(rng)
        self.t_release_lim = t_release_lim
        self.duration_lim = duration_lim
        self.slope_lim = slope_lim
        self.t_drop_lim = t_drop_lim
        self.l_drop_lim = l_drop_lim

        self._sort_dict = {'t_release': lambda task: task.t_release,
                           'duration': lambda task: task.duration,
                           'slope': lambda task: task.slope,
                           't_drop': lambda task: task.t_drop,
                           'l_drop': lambda task: task.l_drop,
                           'priority': lambda task: task.l_drop / task.t_drop
                           }

    def __call__(self, n_tasks, sort_key=None):
        """Randomly generate a list of tasks."""
        t_release = self.rng.uniform(*self.t_release_lim, n_tasks)
        duration = self.rng.uniform(*self.duration_lim, n_tasks)
        slope = self.rng.uniform(*self.slope_lim, n_tasks)
        t_drop = self.rng.uniform(*self.t_drop_lim, n_tasks)
        l_drop = self.rng.uniform(*self.l_drop_lim, n_tasks)

        task_list = [ReluDropTask(*args) for args in zip(t_release, duration, slope, t_drop, l_drop)]

        if sort_key is None:        # FIXME: does sorting belong here or external to class?
            return task_list
        else:
            if type(sort_key) == str:
                sort_key = self._sort_dict[sort_key]
            task_list.sort(key=sort_key)
            return task_list

    @property
    def param_rep_lim(self):
        """Low and high tuples bounding parametric task representations."""
        # return self.t_release_lim, self.duration_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim
        return zip(self.t_release_lim, self.duration_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)


class PermuteTaskGenerator(GenericTaskGenerator):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.rng.permutation(self.tasks)


class DeterministicTaskGenerator(GenericTaskGenerator):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.tasks


def main():
    task_gen = ReluDropGenerator(t_release_lim=(0, 4), duration_lim=(3, 6), slope_lim=(0.5, 2),
                                 t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=None)  # task set generator

    task = task_gen(1)[0]
    a = 1


if __name__ == '__main__':
    main()

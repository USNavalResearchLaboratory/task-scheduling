"""Task objects."""

# TODO: document class attributes, even if identical to init parameters?

import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_rng
from scipy.stats import rv_discrete

np.set_printoptions(precision=2)
plt.style.use('seaborn')


class Generic:
    """
    Generic task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.
    loss_func : function, optional
        Maps execution time to scheduling loss.

    """

    def __init__(self, duration, t_release, loss_func=None):
        self.duration = duration
        self.t_release = t_release

        if callable(loss_func):
            self._loss_func = loss_func

    def __repr__(self):
        return f"Generic(duration: {self.duration:.2f}, release time:{self.t_release:.2f})"

    def __call__(self, t):
        """Loss function versus time."""
        return self._loss_func(t)

    def __eq__(self, other):
        if not isinstance(other, Generic):
            return False

        conditions = [self.duration == other.duration,
                      self.t_release == other.t_release,
                      self._loss_func == other._loss_func]

        return True if all(conditions) else False

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

        Returns
        -------
        matplotlib.lines.Line2D
            Loss function line

        """

        if t_plot is None:
            t_plot = np.arange(*self.plot_lim, 0.01)
        elif t_plot[0] < self.t_release:
            t_plot = t_plot[t_plot >= self.t_release]

        x_lim = t_plot[0], t_plot[-1]
        y_lim = self(x_lim)

        if ax is None:
            _, ax = plt.subplots()

            ax.set(xlabel='t', ylabel='Loss')
            plt.grid(True)
            plt.title(self.__repr__())
        else:
            x_lim_gca, y_lim_gca = ax.get_xlim(), ax.get_ylim()
            x_lim = min(x_lim[0], x_lim_gca[0]), max(x_lim[1], x_lim_gca[1])
            y_lim = min(y_lim[0], y_lim_gca[0]), max(y_lim[1], y_lim_gca[1])

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        plot_data = ax.plot(t_plot, self(t_plot), label=self.__repr__())

        return plot_data


class ReluDrop(Generic):
    """
    Tasks with a rectified linear loss function task and a constant drop penalty.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.
    slope : float
        Function slope between release and drop times. Loss at release time is zero.
    t_drop : float
        Drop time relative to release time.
    l_drop : float
        Constant loss after drop time.

    """

    def __init__(self, duration, t_release, slope, t_drop, l_drop):
        super().__init__(duration, t_release)
        self._slope = slope
        self._t_drop = t_drop
        self._l_drop = l_drop

    def __repr__(self):
        return f"ReluDrop(duration: {self.duration:.2f}, release time:{self.t_release:.2f})"

    def __call__(self, t):
        """Loss function versus time."""
        t = np.asarray(t)[np.newaxis] - self.t_release      # relative time

        loss = self.slope * t
        loss[t < 0] = 0.
        loss[t >= self.t_drop] = self.l_drop
        if loss.size == 1:
            return np.asscalar(loss)
        else:
            return loss.squeeze(axis=0)

    def __eq__(self, other):
        if not isinstance(other, ReluDrop):
            return False

        conditions = [self.duration == other.duration,
                      self.t_release == other.t_release,
                      self.slope == other.slope,
                      self.t_drop == other.t_drop,
                      self.l_drop == other.l_drop]

        return True if all(conditions) else False

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, slope):
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
        """
        Shift the time origin, return any incurred loss, and re-parameterize the task.

        Parameters
        ----------
        t : float
            Positive value to shift the time origin by.

        Returns
        -------
        float
            Loss value of the task at the new time origin, before it is re-parameterized.

        """

        if t <= 0:
            raise ValueError("Shift time must be positive.")

        t_excess = t - self.t_release
        self.t_release = max(0., -t_excess)
        if self.t_release == 0.:
            # Loss is incurred, drop time and loss are updated
            loss_inc = self(t_excess)
            self.t_drop = max(0., self.t_drop - t_excess)
            self.l_drop = self.l_drop - loss_inc
            return loss_inc
        else:
            return 0.   # No loss incurred

    def gen_features(self, *funcs):
        """Generate features from input functions. Defaults to the parametric representation."""
        if len(funcs) > 0:
            return [func(self) for func in funcs]
        else:   # default, return task parameters
            return [self.duration, self.t_release, self.slope, self.t_drop, self.l_drop]

    def summary(self):
        """Print a string listing task parameters."""
        print(f'ReluDrop\n------------\nduration: {self.duration:.2f}'
              f'\nrelease time: {self.t_release:.2f}\nslope: {self.slope:.2f}'
              f'\ndrop time: {self.t_drop:.2f}\ndrop loss: {self.l_drop:.2f}')

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + max(self.duration, self.t_drop) + 1


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


# FIXME FIXME: move KW generator additions to new home.

# %% Task generation objects        # TODO: generalize, add docstrings
class GenericTaskGenerator:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_tasks):
        raise NotImplementedError


class ReluDropGenerator(GenericTaskGenerator):
    """
    Generator of random ReluDropTask objects.

    Parameters
    ----------
    duration_lim : iterable of float
    t_release_lim : iterable of float
    slope_lim : iterable of float
    t_drop_lim : iterable of float
    l_drop_lim : iterable of float
    discrete_flag:

    """

    def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None,
                 discrete_flag=np.array([False, False, False, False, False])):
        super().__init__(rng)
        self.duration_lim = duration_lim
        self.t_release_lim = t_release_lim
        self.slope_lim = slope_lim
        self.t_drop_lim = t_drop_lim
        self.l_drop_lim = l_drop_lim
        self.discrete_flag = discrete_flag

    def __call__(self, n_tasks):
        """Randomly generate a list of tasks."""

        if self.discrete_flag[0]:
            values = np.array(self.duration_lim)
            probabilities = list(np.ones(len(values))/len(values))
            distrib = rv_discrete(values=(range(len(values)), probabilities))
            duration = values[distrib.rvs(size=n_tasks)]
        else:
            duration = self.rng.uniform(*self.duration_lim, n_tasks)

        if self.discrete_flag[1]:
            values = np.array(self.t_release_lim)
            probabilities = list(np.ones(len(values)) / len(values))
            distrib = rv_discrete(values=(range(len(values)), probabilities))
            t_release = values[distrib.rvs(size=n_tasks)]
        else:
            t_release = self.rng.uniform(*self.t_release_lim, n_tasks)

        if self.discrete_flag[2]:
            values = np.array(self.slope_lim)
            probabilities = list(np.ones(len(values)) / len(values))
            distrib = rv_discrete(values=(range(len(values)), probabilities))
            slope = values[distrib.rvs(size=n_tasks)]
        else:
            slope = self.rng.uniform(*self.slope_lim, n_tasks)

        if self.discrete_flag[3]:
            values = np.array(self.t_drop_lim)
            probabilities = list(np.ones(len(values)) / len(values))
            distrib = rv_discrete(values=(range(len(values)), probabilities))
            t_drop = values[distrib.rvs(size=n_tasks)]
        else:
            t_drop = self.rng.uniform(*self.t_drop_lim, n_tasks)

        if self.discrete_flag[4]:
            values = np.array(self.l_drop_lim)
            probabilities = list(np.ones(len(values)) / len(values))
            distrib = rv_discrete(values=(range(len(values)), probabilities))
            l_drop = values[distrib.rvs(size=n_tasks)]
        else:
            l_drop = self.rng.uniform(*self.l_drop_lim, n_tasks)

        return [ReluDropTask(*args) for args in zip(duration, t_release, slope, t_drop, l_drop)]

        # for _ in range(n_tasks):      # FIXME: use yield?
        #     yield ReluDropTask(self.rng.uniform(*self.duration_lim),
        #                        self.rng.uniform(*self.t_release_lim),
        #                        self.rng.uniform(*self.slope_lim),
        #                        self.rng.uniform(*self.t_drop_lim),
        #                        self.rng.uniform(*self.l_drop_lim),
        #                        )

    @property
    def param_rep_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return zip(self.duration_lim, self.t_release_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)


# TODO: update/formalize generators below
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
    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=None)  # task set generator


if __name__ == '__main__':
    main()

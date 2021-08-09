"""Task objects."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#%% Task utilities

def check_task_types(tasks):
    cls_task = tasks[0].__class__
    if all(isinstance(task, cls_task) for task in tasks[1:]):
        return cls_task
    else:
        raise TypeError("All tasks must be of the same type.")


def tasks_to_dataframe(tasks):
    return pd.DataFrame([task.to_series() for task in tasks])


def summarize_tasks(tasks, file=None, **tabulate_kwargs):
    """Create and print a Pandas DataFrame detailing tasks."""
    tabulate_kwargs_ = {'tablefmt': 'github', 'floatfmt': '.3f'}
    tabulate_kwargs_.update(tabulate_kwargs)
    print(tasks_to_dataframe(tasks).to_markdown(**tabulate_kwargs_), file=file)
    # print(tasks_to_dataframe(tasks).to_markdown(tablefmt='github', floatfmt='.3f'))


#%% Task objects

class Base(ABC):
    """
    Base task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.

    """

    param_names = ('duration', 't_release')

    def __init__(self, duration, t_release):
        self.duration = float(duration)
        self.t_release = float(t_release)

    def __repr__(self):
        params_str = ", ".join([f"{name}: {getattr(self, name):.3f}" for name in ("duration", "t_release")])
        # params_str = ", ".join([f"{name}: {getattr(self, name):.3f}" for name in self.param_names])
        return f"{self.__class__.__name__}({params_str})"
        # return f"{self.__class__.__name__}(duration: {self.duration:.3f}, release time: {self.t_release:.3f})"

    @abstractmethod
    def __call__(self, t):
        """Loss function versus time."""
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.params == other.params
        else:
            return NotImplemented

    @property
    def params(self):
        return {name: getattr(self, name) for name in self.param_names}

    def to_series(self, **kwargs):
        return pd.Series(self.params, **kwargs)

    def summary(self, file=None):
        """Print a string listing task parameters."""
        print(self.to_series(name='value').to_markdown(tablefmt='github', floatfmt='.3f'), file=file)
        # summarize_tasks([self], index=False)

        # cls_str = self.__class__.__name__
        #
        # param_str = [f"- {name}: {val}" for name, val in self.params.items()]
        # str_out = '\n'.join([cls_str] + param_str)
        #
        # print(str_out)
        # return str_out

    # def feature_gen(self, *funcs):
    #     """Generate features from input functions. Defaults to the parametric representation."""
    #     if len(funcs) > 0:
    #         return [func(self) for func in funcs]
    #     else:   # default, return task parameters
    #         return list(self.params.values())

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

        if ax is None:
            _, ax = plt.subplots()

            ax.set(xlabel='t', ylabel='Loss')
            plt.title(self.__repr__())

        plot_data = ax.plot(t_plot, self(t_plot), label=self.__repr__())

        return plot_data


class Generic(Base):
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

    param_names = ('duration', 't_release', 'loss_func')

    def __init__(self, duration, t_release, loss_func=None):
        super().__init__(duration, t_release)

        if callable(loss_func):
            self.loss_func = loss_func

    def __call__(self, t):
        """Loss function versus time."""
        return self.loss_func(t)

    # def __eq__(self, other):
    #     if isinstance(other, Generic):
    #         return self.params == other.params and self.loss_func == other.loss_func
    #     else:
    #         return NotImplemented


class Shift(Base):
    @abstractmethod
    def __call__(self, t):
        """Loss function versus time."""
        raise NotImplementedError

    @abstractmethod
    def shift_origin(self, t):
        raise NotImplementedError


class ReluDrop(Shift):
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

    param_names = ('duration', 't_release', 'slope', 't_drop', 'l_drop')

    def __init__(self, duration, t_release, slope, t_drop, l_drop):
        super().__init__(duration, t_release)
        self._slope = float(slope)
        self._t_drop = float(t_drop)
        self._l_drop = float(l_drop)

    def __call__(self, t):
        """Loss function versus time."""
        t = np.asarray(t)[np.newaxis] - self.t_release  # relative time

        loss = self.slope * t
        loss[t < -1e-9] = np.nan
        loss[t >= self.t_drop] = self.l_drop
        if loss.size == 1:
            return loss.item()
        else:
            return loss.squeeze(axis=0)

    # def __eq__(self, other):
    #     if isinstance(other, ReluDrop):
    #         return self.params == other.params
    #     else:
    #         return NotImplemented

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, slope):
        self._check_non_decreasing(slope, self.t_drop, self.l_drop)
        self._slope = slope

    @property
    def t_drop(self):
        return self._t_drop

    @t_drop.setter
    def t_drop(self, t_drop):
        self._check_non_decreasing(self.slope, t_drop, self.l_drop)
        self._t_drop = t_drop

    @property
    def l_drop(self):
        return self._l_drop

    @l_drop.setter
    def l_drop(self, l_drop):
        self._check_non_decreasing(self.slope, self.t_drop, l_drop)
        self._l_drop = l_drop

    @staticmethod
    def _check_non_decreasing(slope, t_drop, l_drop):
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

        if t < 0.:
            raise ValueError("Shift time must be positive.")
        elif t == 0.:
            return 0.

        t_excess = t - self.t_release
        self.t_release = max(0., -t_excess)
        if self.t_release == 0.:
            # Loss is incurred, drop time and loss are updated
            loss_inc = self(t_excess)
            self.t_drop = max(0., self.t_drop - t_excess)
            self.l_drop = self.l_drop - loss_inc
            return loss_inc
        else:
            return 0.  # No loss incurred

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + self.t_drop + 1


#%% Radar tasks

class ReluDropRadar(ReluDrop):
    # param_names = ('duration', 't_release', 'slope', 't_drop', 'l_drop', 't_dwell', 't_revisit')

    def __init__(self, t_dwell, t_revisit, dwell_type=None, revisit_times=None):
        self.t_revisit = t_revisit
        self.dwell_type = dwell_type

        relu_drop_params = {'duration': t_dwell, 't_release': 0, 'slope': 1 / self.t_revisit,
                            't_drop': self.t_revisit + 0.1, 'l_drop': 300}
        super().__init__(**relu_drop_params)

        self.revisit_times = [] if revisit_times is None else list(revisit_times)

    @property
    def count_revisit(self):
        return len(self.revisit_times)

    # def priority(self, t):    # TODO
    #     return self(t)

    @classmethod
    def search(cls, t_dwell, dwell_type):
        if dwell_type == 'HS':
            return cls(t_dwell, t_revisit=2.5, dwell_type=dwell_type)
        elif dwell_type == 'AHS':
            return cls(t_dwell, t_revisit=5, dwell_type=dwell_type)
        else:
            raise ValueError

    @classmethod
    def track(cls, dwell_type):
        if dwell_type == 'low':
            return cls(t_dwell=0.018, t_revisit=4, dwell_type='track_low')
        elif dwell_type == 'med':
            return cls(t_dwell=0.018, t_revisit=2, dwell_type='track_med')
        elif dwell_type == 'high':
            return cls(t_dwell=0.018, t_revisit=1, dwell_type='track_high')
        else:
            raise ValueError

    @classmethod
    def track_notional(cls, dwell_type):
        if dwell_type == 'low':
            return cls(t_dwell=0.020, t_revisit=4, dwell_type='track_low')
        elif dwell_type == 'med':
            return cls(t_dwell=0.020, t_revisit=2, dwell_type='track_med')
        elif dwell_type == 'high':
            return cls(t_dwell=0.020, t_revisit=1, dwell_type='track_high')
        else:
            raise ValueError

    @classmethod
    def from_kinematics(cls, slant_range, rate_range):  # TODO: DRY with search_track?
        if slant_range <= 50:
            return cls.track('high')
        elif slant_range > 50 and abs(rate_range) >= 100:
            return cls.track('med')
        else:
            return cls.track('low')

    @classmethod
    def from_kinematics_notional(cls, slant_range, rate_range):  # TODO: DRY with search_track?
        if slant_range <= 50:
            return cls.track_notional('high')
        elif slant_range > 50 and abs(rate_range) >= 100:
            return cls.track_notional('med')
        else:
            return cls.track_notional('low')

    # @classmethod
    # def dwell_type(cls):
    #     if cls.t_revisit == 2.5:
    #         return 'HS'
    #     elif cls.t_revisit == 5:
    #         return 'AHS'
    #     elif cls.t_revisit == 4:
    #         return 'track_low'
    #     elif cls.t_revisit == 2:
    #         return 'track_med'
    #     elif cls.t_revisit == 1:
    #         return 'track_high'

# class ReluDropSearch(ReluDrop):
#     # param_names = ('duration', 't_release', 'slope', 't_drop', 'l_drop', 't_revisit', 'dwell_type')
#
#     def __init__(self, t_dwell, dwell_type, revisit_times=None):
#         self.dwell_type = dwell_type
#
#         if self.dwell_type == 'HS':
#             self.t_revisit = 2.5
#         elif self.dwell_type == 'AHS':
#             self.t_revisit = 5
#         else:
#             raise ValueError
#
#         relu_drop_params = {'duration': t_dwell, 't_release': 0, 'slope': 1 / self.t_revisit,
#                             't_drop': self.t_revisit + 0.1, 'l_drop': 300}
#         super().__init__(**relu_drop_params)
#
#         self.revisit_times = [] if revisit_times is None else list(revisit_times)
#
#     @property
#     def count_revisit(self):
#         return len(self.revisit_times)
#
#     # def priority(self, t):    # TODO
#     #     return self(t)
#
#
# class ReluDropTrack(ReluDrop):
#     # param_names = ('duration', 't_release', 'slope', 't_drop', 'l_drop', 't_revisit', 'dwell_type')
#
#     def __init__(self, dwell_type, revisit_times=None):
#         self.dwell_type = dwell_type
#
#         if self.dwell_type == 'low':
#             self.t_revisit = 4
#         elif self.dwell_type == 'med':
#             self.t_revisit = 2
#         elif self.dwell_type == 'high':
#             self.t_revisit = 1
#         else:
#             raise ValueError
#
#         relu_drop_params = {'duration': 0.018, 't_release': 0, 'slope': 1 / self.t_revisit,
#                             't_drop': self.t_revisit + 0.1, 'l_drop': 300}
#         super().__init__(**relu_drop_params)
#
#         self.revisit_times = [] if revisit_times is None else list(revisit_times)
#
#     @property
#     def count_revisit(self):
#         return len(self.revisit_times)
#
#     # def priority(self, t):    # TODO
#     #     return self(t)
#
#     @classmethod
#     def from_kinematics(cls, slant_range, rate_range):       # TODO: DRY with search_track?
#         if slant_range <= 50:
#             return cls('high')
#         elif slant_range > 50 and abs(rate_range) >= 100:
#             return cls('med')
#         else:
#             return cls('low')


# class ReluDropRadar(ReluDrop):
#     param_names = ('duration', 't_release', 'slope', 't_drop', 'l_drop', 't_revisit', 'dwell_type')
#
#     def __init__(self, duration, t_release, slope, t_drop, l_drop, t_revisit=None, dwell_type='search'):
#         super().__init__(duration, t_release, slope, t_drop, l_drop)
#         self.t_revisit = [] if t_revisit is None else list(t_revisit)
#         self.dwell_type = dwell_type
#
#     @property
#     def count_revisit(self):
#         return len(self.t_revisit)
#
#     @property
#     def dwell_subtype(self):        # TODO: this logic belongs in the constructor...
#         if self.dwell_type == 'search':
#             if self.slope == 0.4:
#                 return 'HS'
#             elif self.slope == 0.2:
#                 return 'AHS'
#             else:
#                 raise ValueError
#
#         elif self.dwell_type == 'track':
#             if self.slope == 0.25:
#                 return 'low'  # Low Priority Track
#             elif self.slope == 0.5:
#                 return 'med'  # Medium Priority Track
#             elif self.slope == 1:
#                 return 'high'  # High Priority Track
#             else:
#                 raise ValueError
#
#     # def priority(self, t):    # TODO
#     #     return self(t)
#
#     # TODO: make search factory method
#
#     @classmethod
#     def track_from_kinematics(cls, slant_range, rate_range):       # TODO: DRY with search_track?
#         if slant_range <= 50:
#             rate_revisit = 1
#         elif slant_range > 50 and abs(rate_range) >= 100:
#             rate_revisit = 2
#         else:
#             rate_revisit = 4
#
#         params = {'duration': 0.018, 't_release': 0, 'slope': 1 / rate_revisit,
#                   't_drop': rate_revisit + 0.1, 'l_drop': 300, 'dwell_type': 'track'}
#         return cls(**params)


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
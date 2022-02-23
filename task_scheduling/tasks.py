"""Task objects."""

from abc import ABC, abstractmethod
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    def __init__(self, duration, t_release, name=None):
        self.duration = float(duration)
        self.t_release = float(t_release)

        if name is not None:
            self.name = str(name)
        else:
            self.name = rf"{self.__class__.__name__}($d={self.duration:.3f}$, $\rho={self.t_release:.3f}$)"

    @abstractmethod
    def __call__(self, t):
        """Loss function versus time."""
        raise NotImplementedError

    def __str__(self):
        # params_str = rf"$d={self.duration:.3f}$, $\rho={self.t_release:.3f}$"
        # # params_str = ", ".join([f"{name}: {getattr(self, name):.3f}" for name in ("duration", "t_release")])
        # # params_str = ", ".join([f"{name}: {getattr(self, name):.3f}" for name in self.param_names])
        # return f"{self.__class__.__name__}({params_str})"
        return self.name

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

    def summary(self):
        """Print a string listing task parameters."""
        str_ = f"{self.__class__.__name__}"
        str_ += '\n' + self.to_series(name='value').to_markdown(tablefmt='github', floatfmt='.3f')
        return str_

    # def feature_gen(self, *funcs):
    #     """Generate features from input functions. Defaults to the parametric representation."""
    #     if len(funcs) > 0:
    #         return [func(self) for func in funcs]
    #     else:   # default, return task parameters
    #         return list(self.params.values())

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + 1.

    def plot_loss(self, t_plot=None, ax=None):
        """
        Plot loss function.

        Parameters
        ----------
        t_plot : numpy.ndarray, optional
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
            plt.title(self)

        plot_data = ax.plot(t_plot, self(t_plot), label=str(self))

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

    param_names = Base.param_names + ('loss_func',)

    def __init__(self, duration, t_release, loss_func=None, name=None):
        super().__init__(duration, t_release, name)

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
    shift_params = ('t_release',)

    @abstractmethod
    def __call__(self, t):
        """Loss function versus time."""
        raise NotImplementedError

    @abstractmethod
    def shift_origin(self, t):
        raise NotImplementedError


# class ReLUDrop(Shift):
#     """
#     Tasks with a rectified linear loss function and a constant drop penalty.
#
#     Parameters
#     ----------
#     duration : float
#         Time duration of the task.
#     t_release : float
#         Earliest time the task may be scheduled.
#     slope : float
#         Function slope between release and drop times. Loss at release time is zero.
#     t_drop : float
#         Drop time relative to release time.
#     l_drop : float
#         Constant loss after drop time.
#
#     """
#
#     param_names = Base.param_names + ('slope', 't_drop', 'l_drop')
#     shift_params = ('t_release', 't_drop', 'l_drop')
#
#     def __init__(self, duration, t_release, slope, t_drop, l_drop, name=None):
#         super().__init__(duration, t_release, name)
#         self._slope = float(slope)
#         self._t_drop = float(t_drop)
#         self._l_drop = float(l_drop)
#
#         self._check_params()
#
#     def __call__(self, t):
#         """Loss function versus time."""
#
#         t = np.array(t, dtype=float)
#         t -= self.t_release  # relative time
#
#         loss = np.full(t.shape, np.nan)
#         loss[t >= 0] = self.slope * t[t >= 0]
#         loss[t >= self.t_drop] = self.l_drop
#
#         if loss.ndim == 0:
#             loss = loss.item()
#
#         return loss
#
#     @property
#     def slope(self):
#         return self._slope
#
#     @slope.setter
#     def slope(self, slope):
#         # self._check_non_decreasing(slope, self.t_drop, self.l_drop)
#         self._slope = slope
#         self._check_params()
#
#     @property
#     def t_drop(self):
#         return self._t_drop
#
#     @t_drop.setter
#     def t_drop(self, t_drop):
#         # self._check_non_decreasing(self.slope, t_drop, self.l_drop)
#         self._t_drop = t_drop
#         self._check_params()
#
#     @property
#     def l_drop(self):
#         return self._l_drop
#
#     @l_drop.setter
#     def l_drop(self, l_drop):
#         # self._check_non_decreasing(self.slope, self.t_drop, l_drop)
#         self._l_drop = l_drop
#         self._check_params()
#
#     # @staticmethod
#     # def _check_non_decreasing(slope, t_drop, l_drop):
#     #     if l_drop < slope * t_drop:
#     #         raise ValueError("Loss function must be monotonically non-decreasing.")
#
#     def _check_params(self):
#         if self.slope < 0:
#             raise ValueError("Slope must be non-negative.")
#         if self.t_drop < 0:
#             raise ValueError("Drop time must be non-negative.")
#         if self.slope * self.t_drop > self.l_drop:
#             raise ValueError("Loss function must be monotonically non-decreasing.")
#
#     def shift_origin(self, t):
#         """
#         Shift the time origin, return any incurred loss, and re-parameterize the task.
#
#         Parameters
#         ----------
#         t : float
#             Positive value to shift the time origin by.
#
#         Returns
#         -------
#         float
#             Loss value of the task at the new time origin, before it is re-parameterized.
#
#         """
#
#         t_excess = t - self.t_release
#         self.t_release = max(0., -t_excess)
#         if self.t_release == 0.:  # loss is incurred, drop time and loss are updated
#             loss_inc = self(t_excess)
#             self._t_drop = max(0., self._t_drop - t_excess)
#             self._l_drop = self._l_drop - loss_inc
#             return loss_inc
#         else:
#             return 0.  # no loss incurred
#
#     @property
#     def plot_lim(self):
#         """2-tuple of limits for automatic plotting."""
#         return self.t_release, self.t_release + self.t_drop + 1.


class PiecewiseLinear(Shift):
    param_names = Base.param_names + ('l_release', 'slope', 'corners')
    shift_params = Shift.shift_params  # TODO: Add shift params. Handle `list` parameters for `space` shifts?!?

    prune = True

    def __init__(self, duration, t_release=0., l_release=0., slope=1., corners=(), name=None):
        super().__init__(duration, t_release, name)
        self.l_release = float(l_release)
        self.slope = float(slope)
        self.corners = corners

        self._check_non_decreasing()

    def __call__(self, t):
        """Loss function versus time."""

        t = np.array(t, dtype=float)
        t -= self.t_release  # relative time

        loss = np.full(t.shape, np.nan)
        loss[t >= 0] = self.l_release + self.slope * t[t >= 0]
        for t_c, l_c, s_c in self.corners:
            loss[t >= t_c] = l_c + s_c * (t[t >= t_c] - t_c)

        if loss.ndim == 0:
            loss = loss.item()

        return loss

    @property
    def corners(self):
        return self._corners

    @corners.setter
    def corners(self, val):
        val = list(map(list, val))
        val = sorted(val, key=itemgetter(0))  # sort by time
        for i, c in enumerate(val):
            if len(c) == 2:
                t = c[0]
                if i == 0:
                    c.insert(1, self.l_release + self.slope * t)
                else:
                    t_prev, s_prev, l_prev = val[i-1]
                    c.insert(1, l_prev + s_prev * (t - t_prev))

        self._corners = val

    def _check_non_decreasing(self):  # TODO: integrate with param setters?
        if self.l_release < 0.:
            raise ValueError("Release loss must be non-negative.")
        if self.slope < 0.:
            raise ValueError("Slope must be non-negative.")

        for i, c in enumerate(self.corners):
            t_c, l_c, s_c = c
            if self.prune:
                if t_c <= 0.:
                    raise ValueError("Relative corner times must be positive.")
            else:
                if t_c < 0.:
                    raise ValueError("Relative corner times must be non-negative.")
            if l_c < 0.:
                raise ValueError("Corner losses must be non-negative.")
            if s_c < 0.:
                raise ValueError("Corner slopes must be non-negative.")

            if i == 0:
                l_d = self.l_release + self.slope * t_c
            else:
                t_prev, l_prev, s_prev = self.corners[i - 1]
                l_d = l_prev + s_prev * (t_c - t_prev)
            if l_c < l_d:
                raise ValueError(f"Loss decreases from {l_d} to {l_c} at discontinuity.")

    def shift_origin(self, t):
        t_excess = t - self.t_release
        self.t_release = max(0., -t_excess)
        if self.t_release == 0.:  # loss is incurred, drop time and loss are updated
            loss_inc = self(t_excess)
            self.l_release = max(0., self.l_release - loss_inc)
            for i, c in enumerate(self.corners):
                c[0] = max(0., c[0] - t_excess)
                c[1] = max(0., c[1] - loss_inc)

                if c[0] == 0.:  # zero out unused slope
                    if i == 0:
                        self.slope = 0.
                    else:
                        self.corners[i - 1][2] = 0.

            if self.prune:
                self.prune_corners()

            return loss_inc
        else:
            return 0.  # no loss incurred

    def prune_corners(self):
        i_keep = 0
        for i, c in enumerate(reversed(self.corners)):
            if c[0] == 0.:
                i_keep = len(self.corners) - i
                self.l_release, self.slope = c[1:]
                break
        self.corners = self.corners[i_keep:]

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        t_1 = self.t_release
        if bool(self.corners) and self.corners[-1][0] > 0.:
            t_1 += self.corners[-1][0] * (1 + plt.rcParams['axes.xmargin'])
        else:
            t_1 += 1.
        return self.t_release, t_1


class ReLU(PiecewiseLinear):
    param_names = Base.param_names + ('slope',)
    shift_params = Shift.shift_params

    def __init__(self, duration, t_release, slope, name=None):
        super().__init__(duration, t_release, 0., slope, [], name)


class ReLUDrop(PiecewiseLinear):
    param_names = Base.param_names + ('slope', 't_drop', 'l_drop')
    shift_params = Shift.shift_params + ('t_drop', 'l_drop')

    prune = False

    def __init__(self, duration, t_release, slope, t_drop, l_drop, name=None):
        super().__init__(duration, t_release, 0., slope, [[t_drop, l_drop, 0.]], name)

    @property
    def t_drop(self):
        return self.corners[0][0]

    @t_drop.setter
    def t_drop(self, val):
        self.corners[0][0] = val

    @property
    def l_drop(self):
        return self.corners[0][1]

    @l_drop.setter
    def l_drop(self, val):
        self.corners[0][1] = val


# class Radar(ReLUDrop):
#     def __init__(self, duration, t_release, t_revisit, dwell_type=None):
#         self.t_revisit = t_revisit
#         self.dwell_type = dwell_type
#
#         relu_drop_params = dict(
#             slope=1 / self.t_revisit,
#             t_drop=self.t_revisit + 0.1,
#             l_drop=300,
#         )
#         super().__init__(duration, t_release, **relu_drop_params)
#
#     @classmethod
#     def search(cls, t_release, dwell_type):
#         t_dwell = 0.36
#         # t_revisit = dict(HS=2.5, AHS=5)[dwell_type]
#         t_revisit = dict(HS=5.88, AHS=11.76)[dwell_type]
#         return cls(t_dwell, t_release, t_revisit, dwell_type)
#
#     @classmethod
#     def track(cls, t_release, dwell_type):
#         # t_dwell = 0.18
#         # t_revisit = dict(low=4, med=2, high=1)[dwell_type]
#         t_dwell = 0.36
#         t_revisit = dict(low=1, high=.5)[dwell_type]
#         return cls(t_dwell, t_release, t_revisit, 'track_' + dwell_type)
#
#     # @classmethod
#     # def from_kinematics(cls, slant_range, rate_range):
#     #     if slant_range <= 50:
#     #         return cls.track('high')
#     #     elif slant_range > 50 and abs(rate_range) >= 100:
#     #         return cls.track('med')
#     #     else:
#     #         return cls.track('low')

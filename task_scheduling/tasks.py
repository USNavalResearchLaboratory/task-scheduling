"""Task classes."""

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
        The earliest time the task may be scheduled.
    name : str, optional
        Name of the task.

    """

    param_names = ("duration", "t_release")

    def __init__(self, duration, t_release, name=None):
        self.duration = float(duration)
        self.t_release = float(t_release)

        if name is not None:
            self.name = str(name)
        else:
            self.name = (
                rf"{self.__class__.__name__}($d={self.duration:.3f}$, $\rho={self.t_release:.3f}$)"
            )

    @abstractmethod
    def __call__(self, t):
        """
        Loss function versus time.

        Parameters
        ----------
        t : float
            Execution time.

        Returns
        -------
        float
            Execution loss.

        """
        raise NotImplementedError

    def __str__(self):
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
        str_ += "\n" + self.to_series(name="value").to_markdown(tablefmt="github", floatfmt=".3f")
        return str_

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        return self.t_release, self.t_release + self.duration

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
            t_plot = np.arange(*self.plot_lim, 1e-3)

        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel="t", ylabel="Loss", title=str(self))

        return ax.plot(t_plot, self(t_plot), label=str(self))


class ShiftMixin(ABC):
    def shift(self, t):
        """
        Shift the release time and loss function.

        Parameters
        ----------
        t : float
            Temporal shift to advance the task.

        """
        self.t_release -= t
        self._shift(t)

    @abstractmethod
    def _shift(self, t):
        raise NotImplementedError

    def reparam(self, t):
        """
        Reparameterize the task, return any incurred loss.

        Parameters
        ----------
        t : float
            Time at which to evaluate the task.

        Returns
        -------
        float
            Partial execution loss.

        """
        if self.t_release < t:
            loss_inc = self(t)
            self._reparam(t)
            return loss_inc
        else:
            return 0.0

    @abstractmethod
    def _reparam(self, t):
        raise NotImplementedError

    @staticmethod
    def shift_param_lims(param_lims, ch_avail_lim, n_tasks):
        param_lims["t_release"] = (0.0, param_lims["t_release"][1])
        return param_lims


class Generic(Base):
    """
    Generic task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        The earliest time the task may be scheduled.
    loss_func : function, optional
        Maps execution time to scheduling loss.

    """

    param_names = Base.param_names + ("loss_func",)

    def __init__(self, duration, t_release, loss_func=None, name=None):
        super().__init__(duration, t_release, name)

        if callable(loss_func):
            self.loss_func = loss_func

    def __call__(self, t):
        """
        Loss function versus time.

        Parameters
        ----------
        t : float
            Execution time.

        Returns
        -------
        float
            Execution loss.

        """
        return self.loss_func(t)


class PiecewiseLinear(ShiftMixin, Base):
    """
    Task with a piecewise linear loss function.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        The earliest time the task may be scheduled.
    corners : Sequence of Sequence, optional
        Each element is a 3-tuple of the corner time (relative to release time), loss,
        and proceeding slope.
    name : str, optional
        Name of the task.

    """

    param_names = Base.param_names + ("corners",)
    # TODO: Add shift params. Handle `list` parameters for `space` shifts?!?

    prune = True

    def __init__(self, duration, t_release=0.0, corners=(), name=None):
        super().__init__(duration, t_release, name)
        self.corners = corners

        self._check_non_decreasing()

    def __call__(self, t):
        """
        Loss function versus time.

        Parameters
        ----------
        t : float
            Execution time.

        Returns
        -------
        float
            Execution loss.

        """
        t = np.array(t, dtype=float)
        t -= self.t_release  # relative time

        loss = np.full(t.shape, np.nan)

        loss[t >= 0] = 0.0
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
            if len(c) == 2:  # interpret as time and slope, calculate loss for continuity
                t = c[0]
                if i == 0:
                    c.insert(1, 0.0)
                else:
                    t_prev, l_prev, s_prev = val[i - 1]
                    c.insert(1, l_prev + s_prev * (t - t_prev))

        self._corners = val

    def _check_non_decreasing(self):  # TODO: integrate with param setters?
        for i, c in enumerate(self.corners):
            t_c, l_c, s_c = c
            if t_c < 0.0:
                raise ValueError("Relative corner times must be non-negative.")
            if l_c < 0.0:
                raise ValueError("Corner losses must be non-negative.")
            if s_c < 0.0:
                raise ValueError("Corner slopes must be non-negative.")

            if i == 0:
                l_d = 0.0
            else:
                t_prev, l_prev, s_prev = self.corners[i - 1]
                l_d = l_prev + s_prev * (t_c - t_prev)
            if l_c < l_d:
                raise ValueError(f"Loss decreases from {l_d} to {l_c} at discontinuity.")

    def _prune_corners(self):
        corners = np.array(self.corners)
        times = corners[:, 0]
        t_u = np.unique(times)
        idx = []
        if len(t_u) < len(self.corners):
            for t in t_u:
                idx.append(np.flatnonzero(times == t)[-1])
            self._corners = corners[idx].tolist()

            if self._corners == [[0.0, 0.0, 0.0]]:  # remove uninformative corner
                self._corners = []

    @property
    def plot_lim(self):
        """2-tuple of limits for automatic plotting."""
        t_1 = self.t_release
        if bool(self.corners) and self.corners[-1][0] > 0.0:
            t_1 += self.corners[-1][0] * (1 + plt.rcParams["axes.xmargin"])
        else:
            t_1 += self.duration
        return self.t_release, t_1

    def _shift(self, t):
        pass

    def _reparam(self, t):
        loss = self(t)
        for c in self.corners:
            c[0] = max(0.0, c[0] - (t - self.t_release))
            c[1] = max(0.0, c[1] - loss)

            # if not self.prune and c[0] == 0. and i >= 1:  # zero out unused slope
            #     self.corners[i - 1][2] = 0.

        if self.prune:
            self._prune_corners()

        self.t_release = t


class Linear(PiecewiseLinear):
    """
    Task with a linear loss function.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        The earliest time the task may be scheduled.
    slope : float, optional
    name : str, optional
        Name of the task.

    """

    param_names = Base.param_names + ("slope",)

    prune = False

    def __init__(self, duration, t_release=0.0, slope=1.0, name=None):
        super().__init__(duration, t_release, [[0.0, 0.0, slope]], name)

    @property
    def slope(self):
        return self.corners[0][2]

    @slope.setter
    def slope(self, val):
        self.corners[0][2] = val


class LinearDrop(PiecewiseLinear):
    """
    Task with a piecewise linear loss function leading to a constant value.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        The earliest time the task may be scheduled.
    slope : float, optional
    t_drop : float, optional
        Drop time, relative to release time.
    l_drop : float, optional
        Drop loss.
    name : str, optional
        Name of the task.

    """

    param_names = Base.param_names + ("slope", "t_drop", "l_drop")

    prune = False

    def __init__(self, duration, t_release=0.0, slope=1.0, t_drop=1.0, l_drop=None, name=None):
        corners = [[0.0, 0.0, slope]]
        if l_drop is not None:
            corners.append([t_drop, l_drop, 0.0])
        else:
            corners.append([t_drop, 0.0])
        super().__init__(duration, t_release, corners, name)

    @property
    def slope(self):
        return self.corners[0][2]

    @slope.setter
    def slope(self, val):
        self.corners[0][2] = val

    @property
    def t_drop(self):
        return self.corners[1][0]

    @t_drop.setter
    def t_drop(self, val):
        self.corners[1][0] = val

    @property
    def l_drop(self):
        return self.corners[1][1]

    @l_drop.setter
    def l_drop(self, val):
        self.corners[1][1] = val

    @staticmethod
    def shift_param_lims(param_lims, ch_avail_lim, n_tasks):
        new_lims = super(LinearDrop, LinearDrop).shift_param_lims(param_lims, ch_avail_lim, n_tasks)
        for param in ("t_drop", "l_drop"):
            new_lims[param] = (0.0, param_lims[param][1])
        return new_lims


class Exponential(ShiftMixin, Base):
    """
    Task with an exponential loss function.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        The earliest time the task may be scheduled.
    a : float, optional
        Multiplicative constant.
    b : float, optional
        Exponent base.
    name : str, optional
        Name of the task.

    """

    param_names = Base.param_names + ("a", "b")

    def __init__(self, duration, t_release=0.0, a=1.0, b=np.e, name=None):
        super().__init__(duration, t_release, name)
        self.a = float(a)
        self.b = float(b)

        if not a > 0.0:
            raise ValueError("a must be positive.")
        if not b >= 1.0:
            raise ValueError("b must be equal to or greater than 1.")

    def __call__(self, t):
        """Loss function versus time."""
        t = np.array(t, dtype=float)
        t -= self.t_release  # relative time

        loss = np.full(t.shape, np.nan)

        loss[t >= 0] = self.a * (self.b ** t[t >= 0] - 1)

        if loss.ndim == 0:
            loss = loss.item()

        return loss

    def _shift(self, t):
        pass

    def _reparam(self, t):
        self.a *= self.b ** (t - self.t_release)
        self.t_release = t

    @staticmethod
    def shift_param_lims(param_lims, ch_avail_lim, n_tasks):
        new_lims = super(Exponential, Exponential).shift_param_lims(
            param_lims, ch_avail_lim, n_tasks
        )

        max_start = max(ch_avail_lim[1], param_lims["t_release"][1])
        max_ch_avail = max_start + n_tasks * param_lims["duration"][1]
        max_shift = max_ch_avail - param_lims["t_release"][0]

        high = param_lims["a"][1] * param_lims["b"][1] ** max_shift
        new_lims["a"] = (param_lims["a"][0], high)
        return new_lims

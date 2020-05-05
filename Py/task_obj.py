import numpy as np


class TasksRRM:
    """Generic task objects."""

    def __init__(self, t_release, duration, loss_fcn):
        self.t_release = t_release
        self.duration = duration
        self.loss_fcn = loss_fcn

    @classmethod
    def lin_drop(cls, t_release, duration, w, t_drop, l_drop):
        return cls(t_release, duration, loss_lin_drop(t_release, w, t_drop, l_drop))


def loss_lin_drop(t_release, w, t_drop, l_drop):
    """Linearly increasing loss with constant drop loss."""

    if l_drop < w * (t_drop - t_release):
        raise ValueError("Function is not monotonically non-decreasing.")

    def loss_fcn(t):
        if t < t_release:
            loss = np.inf
        elif (t >= t_release) and (t < t_drop):
            loss = w*(t - t_release)
        else:
            loss = l_drop

        return loss

    return np.vectorize(loss_fcn)

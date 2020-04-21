import numpy as np


class TasksRRM:
    def __init__(self, t_start, duration, loss_fcn):
        self.t_start = t_start
        self.duration = duration
        self.loss_fcn = loss_fcn

    @classmethod
    def lin_drop(cls, t_start, duration, w, t_drop, l_drop):
        return cls(t_start, duration, loss_lin_drop(t_start, w, t_drop, l_drop))


def loss_lin_drop(t_start, w, t_drop, l_drop):
    """Linearly increasing loss with constant drop loss."""

    if l_drop < w * (t_drop - t_start):
        raise ValueError("Function is not monotonically non-decreasing.")

    def loss_fcn(t):
        if t < t_start:
            loss = np.inf
        elif (t >= t_start) and (t < t_drop):
            loss = w*(t-t_start)
        else:
            loss = l_drop

        return loss

    return np.vectorize(loss_fcn)

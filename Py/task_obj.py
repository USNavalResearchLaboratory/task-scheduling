"""Task objects."""

import numpy as np


class TaskRRM:
    """Generic task objects.

    Parameters
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.
    loss_fcn : function
        Function returning losses for all elements of a time array.

    Attributes
    ----------
    duration : float
        Time duration of the task.
    t_release : float
        Earliest time the task may be scheduled.
    loss_fcn : function
        Function returning losses for all elements of a time array.

    """

    def __init__(self, duration, t_release, loss_fcn):
        self.duration = duration
        self.t_release = t_release
        self.loss_fcn = loss_fcn

    @classmethod
    def relu_drop(cls, duration, t_release, w, t_drop, l_drop):
        """Generates a task object with a rectified linear loss function with a constant drop penalty.

            See documentation of task_obj.TaskRRM and task_obj.loss_relu_drop for parameter descriptions.

        """

        return cls(duration, t_release, loss_relu_drop(t_release, w, t_drop, l_drop))


def loss_relu_drop(t_release, w, t_drop, l_drop):
    """Generates a rectified linear loss function with a constant drop penalty.

    Parameters
    ----------
    t_release : float
        Earliest time the task may be scheduled. Loss at t_release is zero.
    w : float
        Function slope between release and drop times.
    t_drop : float
        Drop time.
    l_drop : float
        Constant loss after drop time.

    Returns
    -------
    function

    """

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

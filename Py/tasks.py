"""Task objects."""

import numpy as np

rng_default = np.random.default_rng()


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

    def __init__(self, duration, t_release):
        self.duration = duration
        self.t_release = t_release
        # self.loss_fcn = loss_fcn

    def loss_fcn(self, t):
        raise NotImplementedError

    # @classmethod
    # def relu_drop(cls, duration, t_release, slope, t_drop, l_drop):
    #     """Generates a task object with a rectified linear loss function with a constant drop penalty.
    #
    #         See documentation of task_obj.TaskRRM and task_obj.loss_relu_drop for parameter descriptions.
    #
    #     """
    #
    #     return cls(duration, t_release, loss_relu_drop(t_release, slope, t_drop, l_drop),
    #                slope=slope, t_drop=t_drop, l_drop=l_drop)


class ReluDropTask(TaskRRM):
    def __init__(self, duration, t_release, slope, t_drop, l_drop):
        super().__init__(duration, t_release)
        self.slope = slope
        self.t_drop = t_drop
        self.l_drop = l_drop

        if l_drop < slope * (t_drop - t_release):
            raise ValueError("Function is not monotonically non-decreasing.")

    def __repr__(self):
        return f"ReluDropTask(duration: {self.duration:.3f}, release time:{self.t_release:.3f})"

    def loss_fcn(self, t):
        t = np.asarray(t)[np.newaxis]
        loss = self.slope * (t - self.t_release)
        loss[t < self.t_release] = np.inf
        loss[t >= self.t_drop] = self.l_drop

        return loss.squeeze(axis=0)



def loss_relu_drop(t_release, slope, t_drop, l_drop):
    """Generates a rectified linear loss function with a constant drop penalty.

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

    """

    if l_drop < slope * (t_drop - t_release):
        raise ValueError("Function is not monotonically non-decreasing.")

    def loss_fcn(t):
        if t < t_release:
            loss = np.inf
        elif (t >= t_release) and (t < t_drop):
            loss = slope*(t - t_release)
        else:
            loss = l_drop

        return loss

    return np.vectorize(loss_fcn)


# %% Task generation objects        # TODO: docstrings
class TaskRRMGenerator:
    def __init__(self, rng=rng_default):
        self.rng = rng
        # TODO: state for non-stationary environments?

    def rand_tasks(self, n_tasks, return_params=False):
        raise NotImplementedError


class ReluDropGenerator(TaskRRMGenerator):  # TODO: generalize
    def __init__(self, rng=rng_default):
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

        # tasks = [TaskRRM.relu_drop(*args) for args in _params]
        tasks = [ReluDropTask(*args) for args in _params]

        if return_params:
            return tasks, params
        else:
            return tasks

import numpy as np


def check_valid(tasks, t_ex, ch_ex):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of GenericTask
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    Raises
    -------
    ValueError
        If tasks overlap in time.

    """

    # if np.isnan(t_ex).any():
    #     raise ValueError("All tasks must be scheduled.")

    for ch in np.unique(ch_ex):
        tasks_ch = np.asarray(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch)):
            if t_ex_ch[n_1] < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                if t_ex_ch[n_1] - tasks_ch[n_2].duration + 1e-12 < t_ex_ch[n_2] < t_ex_ch[n_1] \
                        + tasks_ch[n_1].duration - 1e-12:
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def eval_loss(tasks, t_ex):
    """
    Evaluate scheduling loss.

    Parameters
    ----------
    tasks : list of GenericTask
    t_ex : ndarray
        Task execution times.

    Returns
    -------
    float
        Total loss of scheduled tasks.

    """

    l_ex = 0
    for task, t_ex in zip(tasks, t_ex):
        l_ex += task(t_ex)

    return l_ex



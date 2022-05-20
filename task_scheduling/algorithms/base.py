"""Classic scheduling algorithms."""

from task_scheduling.nodes import ScheduleNode, ScheduleNodeBound


def branch_bound(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNodeBound(tasks, ch_avail, rng=rng)
    node_best = node.branch_bound(inplace=False, verbose=verbose)

    return node_best.sch  # optimal


def branch_bound_priority(tasks, ch_avail, priority_func=None, heuristic=None, verbose=False):
    """
    Branch-and-Bound with priority queueing and variable heuristic.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    priority_func : callable, optional
        Key function that maps `ScheduleNode` objects to priority values. Defaults to negative lower bound.
    heuristic : callable, optional
        Uses a partial node to generate a complete sequence node.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNodeBound(tasks, ch_avail)
    node_best = node.branch_bound_priority(priority_func, heuristic, False, verbose)

    return node_best.sch  # optimal


def mcts(
    tasks,
    ch_avail,
    max_runtime=float("inf"),
    max_rollouts=None,
    c_explore=0.0,
    th_visit=0,
    verbose=False,
    rng=None,
):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    max_runtime : float, optional
        Allotted algorithm runtime.
    max_rollouts : int, optional
        Maximum number of rollouts allowed.
    c_explore : float, optional
        Exploration weight. Higher values prioritize less frequently visited notes.
    th_visit : int, optional
        Nodes with up to this number of visits will select children using the `expansion` method.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail, rng=rng)
    node = node.mcts(max_runtime, max_rollouts, c_explore, th_visit, inplace=False, verbose=verbose)

    return node.sch


def random_sequencer(tasks, ch_avail, rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail, rng=rng)
    node.roll_out()

    return node.sch


def priority_sorter(tasks, ch_avail, func, reverse=True):
    """
    Sort tasks based on function value.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    func : callable
        Returns scalar value for task priority.
    reverse : bool, optional
        If `True`, tasks are scheduled in order of decreasing priority value.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail)
    node.priority_sorter(func, reverse)

    return node.sch


def earliest_release(tasks, ch_avail):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail)
    node.earliest_release()

    return node.sch


def earliest_drop(tasks, ch_avail):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail)
    node.earliest_drop()

    return node.sch


def brute_force(tasks, ch_avail, verbose=False):
    """
    Exhaustively search all complete sequences.

    Parameters
    ----------
    tasks : Collection of task_scheduling.tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    numpy.ndarray
        Task schedule.

    """

    node = ScheduleNode(tasks, ch_avail)
    node_best = node.brute_force(inplace=False, verbose=verbose)

    return node_best.sch

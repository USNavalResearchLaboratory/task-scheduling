from task_scheduling.tree_search import TreeNodeBound, TreeNode


def branch_bound(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNodeBound(tasks, ch_avail, rng=rng)
    node_best = node.branch_bound(inplace=False, verbose=verbose)

    return node_best.t_ex, node_best.ch_ex  # optimal


def branch_bound_priority(tasks, ch_avail, priority_func=None, heuristic=None, verbose=False):
    """
    Branch-and-Bound with priority queueing and variable heuristic.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    priority_func : callable, optional
        Key function that maps `TreeNode` objects to priority values. Defaults to negative lower bound.
    heuristic : callable, optional
        Uses a partial node to generate a complete sequence node.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNodeBound(tasks, ch_avail)
    node_best = node.branch_bound_priority(priority_func, heuristic, False, verbose)

    return node_best.t_ex, node_best.ch_ex  # optimal


def mcts(tasks, ch_avail, runtime, c_explore=0., visit_threshold=0, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    runtime : float
            Allotted algorithm runtime.
    c_explore : float, optional
        Exploration weight. Higher values prioritize less frequently visited notes.
    visit_threshold : int, optional
        Nodes with up to this number of visits will select children using the `expansion` method.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail, rng=rng)
    node = node.mcts(runtime, c_explore, visit_threshold, inplace=False, verbose=verbose)

    return node.t_ex, node.ch_ex


def mcts_v1(tasks, ch_avail, runtime, c_explore=1., verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    runtime : float
            Allotted algorithm runtime.
    c_explore : float, optional
        Exploration weight. Higher values prioritize unexplored tree nodes.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail, rng=rng)
    node = node.mcts_v1(runtime, c_explore, inplace=False, verbose=verbose)

    return node.t_ex, node.ch_ex

# def mcts(tasks, ch_avail, n_mc=1, c_explore=0., visit_threshold=0, verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : Sequence of task_scheduling.tasks.Base
#     ch_avail : Sequence of float
#         Channel availability times.
#     n_mc : int, optional
#         Number of complete sequences evaluated.
#     c_explore : float, optional
#         Exploration weight. Higher values prioritize less frequently visited notes.
#     visit_threshold : int, optional
#         Nodes with up to this number of visits will select children using the `expansion` method.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng : int or RandomState or Generator, optional
#         NumPy random number generator or seed. Instance RNG if None.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
#
#     node = TreeNode(tasks, ch_avail, rng=rng)
#     node = node.mcts(n_mc, c_explore, visit_threshold, inplace=False, verbose=verbose)
#
#     return node.t_ex, node.ch_ex
#
#
# def mcts_v1(tasks, ch_avail, n_mc=1, c_explore=1., verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : Sequence of task_scheduling.tasks.Base
#     ch_avail : Sequence of float
#         Channel availability times.
#     n_mc : int, optional
#         Number of roll-outs performed.
#     c_explore : float, optional
#         Exploration weight. Higher values prioritize unexplored tree nodes.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng : int or RandomState or Generator, optional
#         NumPy random number generator or seed. Instance RNG if None.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
#
#     node = TreeNode(tasks, ch_avail, rng=rng)
#     node = node.mcts_v1(n_mc, c_explore, inplace=False, verbose=verbose)
#
#     return node.t_ex, node.ch_ex


def random_sequencer(tasks, ch_avail, rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail, rng=rng)
    node.roll_out()

    return node.t_ex, node.ch_ex


def earliest_release(tasks, ch_avail):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail)
    node.earliest_release()

    return node.t_ex, node.ch_ex


def earliest_drop(tasks, ch_avail):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail)
    node.earliest_drop()

    return node.t_ex, node.ch_ex


def brute_force(tasks, ch_avail, verbose=False):
    """
    Exhaustively search all complete sequences.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail)
    node_best = node.brute_force(inplace=False, verbose=verbose)

    return node_best.t_ex, node_best.ch_ex

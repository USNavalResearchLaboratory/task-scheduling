from numbers import Integral


def mcts_orig(tasks, ch_avail, n_mc, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
        Channel availability times.
    n_mc : int or Iterable of int
        Number of Monte Carlo roll-outs per task.
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
    node_best = node.roll_out(inplace=False)

    n_tasks = len(tasks)
    if isinstance(n_mc, Integral):
        n_mc = n_tasks * [int(n_mc)]

    for n in range(n_tasks):
        if verbose:
            print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')

        # Perform Roll-outs
        for _ in range(n_mc[n]):
            node_mc = node.roll_out(inplace=False)

            if node_mc.l_ex < node_best.l_ex:  # Update best node
                node_best = node_mc

        # Assign next task from earliest available channel
        node.seq_append(node_best.seq[n], check_valid=False)

    return node_best.t_ex, node_best.ch_ex


def branch_bound_lo(tasks, ch_avail, verbose=False, rng=None):

    node = TreeNodeBoundLo(tasks, ch_avail, rng=rng)
    node_best = node.branch_bound(inplace=False, verbose=verbose)

    return node_best.t_ex, node_best.ch_ex  # optimal
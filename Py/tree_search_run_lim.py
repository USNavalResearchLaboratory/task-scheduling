"""Multi-channel Tree Search objects and algorithms."""

from time import perf_counter
from math import floor, factorial
import numpy as np
from util.generic import check_rng

from tasks import ReluDropGenerator
from tree_search import TreeNode, TreeNodeBound, SearchNode


def branch_bound(tasks: list, ch_avail: list, max_runtime=float('inf'), verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    max_runtime : float
        Allotted algorithm runtime.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    stack = [TreeNodeBound()]  # Initialize Stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution

    # Iterate
    run = True
    while run and len(stack) > 0:
        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < node_best.l_ex:  # New node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

            # Check run conditions
            if perf_counter() - t_run >= max_runtime:
                run = False
                break

        if verbose:
            # progress = 1 - sum([math.factorial(len(node.seq_rem)) for node in stack]) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {node_best.l_ex:.3f}', end='\r')

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    return t_ex, ch_ex


def mcts_orig(tasks: list, ch_avail: list, max_runtime=float('inf'), n_mc=None, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    n_mc : int or list of int
        Number of Monte Carlo roll-outs per task.
    max_runtime : float
        Allotted algorithm runtime.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    node = TreeNode()
    node_best = node.roll_out(do_copy=True)

    n_tasks = len(tasks)
    if n_mc is None:
        n_mc = [floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)]
    elif type(n_mc) == int:
        n_mc = n_tasks * [n_mc]

    run = True
    for n in range(n_tasks):
        if verbose:
            print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')

        # Perform Roll-outs
        for _ in range(n_mc[n]):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_best.l_ex:  # Update best node
                node_best = node_mc

            # Check run conditions
            if perf_counter() - t_run >= max_runtime:
                run = False
                break

        if not run:     # TODO: refactor nested break to func w/ return?
            break

        # Assign next task from earliest available channel
        node._seq_extend([node_best.seq[n]])

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex

    return t_ex, ch_ex


def mcts(tasks: list, ch_avail: list, max_runtime, verbose=False):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    max_runtime : float
        Allotted algorithm runtime.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    # TODO: add exploration/exploitation input control.
    # TODO: add early termination for completed search.

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    # TreeNode._rng = check_rng(rng)

    SearchNode.n_tasks = len(tasks)
    tree = SearchNode()

    loss_min = float('inf')
    while perf_counter() - t_run < max_runtime:
        if verbose:
            print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')

        seq = tree.simulate()   # Roll-out a complete sequence
        node = TreeNode(seq)    # Evaluate execution times and channels, total loss

        loss = node.l_ex
        tree.backup(seq, loss)  # Update search tree from leaf sequence to root

        if loss < loss_min:
            node_best, loss_min = node, loss

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex

    return t_ex, ch_ex


def random_sequencer(tasks: list, ch_avail: list, max_runtime=float('inf'), rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    max_runtime : float
        Allotted algorithm runtime.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    node = TreeNode().roll_out(do_copy=True)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    runtime = perf_counter() - t_run
    if runtime >= max_runtime:
        raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

    return t_ex, ch_ex


def earliest_release(tasks: list, ch_avail: list, max_runtime=float('inf'), do_swap=False):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    max_runtime : float
        Allotted algorithm runtime.
    do_swap : bool
        Enables task swapping

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_release for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    runtime = perf_counter() - t_run
    if runtime >= max_runtime:
        raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

    return t_ex, ch_ex


def earliest_drop(tasks: list, ch_avail: list, max_runtime=float('inf'), do_swap=False):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : list of ReluDropTask
    ch_avail : list of float
        Channel availability times.
    max_runtime : float
        Allotted algorithm runtime.
    do_swap : bool
        Enables task swapping.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_run = perf_counter()

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_drop for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    runtime = perf_counter() - t_run
    if runtime >= max_runtime:
        raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

    return t_ex, ch_ex


def main():
    n_tasks = 8
    n_channels = 2

    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=None)  # task set generator

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 2, n_ch)

    tasks = task_gen.rand_tasks(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(None)

    # node = TreeNode([3, 1])
    # node.seq = [3, 1, 4]

    # t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True, rng=None)

    SearchNode.n_tasks = n_tasks

    node = SearchNode()
    child = node.select_child()
    leaf = node.simulate()
    pass

    t_ex, ch_ex = mcts_orig(tasks, ch_avail, n_mc=[1000 for n in range(n_tasks, 0, -1)], verbose=False)
    print(t_ex)
    t_ex, ch_ex = mcts(tasks, ch_avail, verbose=False)
    print(t_ex)


if __name__ == '__main__':
    main()

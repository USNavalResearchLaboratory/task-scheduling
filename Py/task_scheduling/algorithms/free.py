from collections import deque
from copy import deepcopy
from numbers import Integral

import numpy as np

from task_scheduling.tree_search import TreeNodeBound, TreeNode, SearchNode

# from sequence2schedule import FlexDARMultiChannelSequenceScheduler


def branch_bound(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
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

    # TODO: different search strategies? pre-sort?

    stack = deque([TreeNodeBound(tasks, ch_avail, rng=rng)])        # initialize stack
    node_best = stack[0].roll_out(inplace=False)  # roll-out initial solution

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # extract node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < node_best.l_ex:  # new node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(inplace=False)  # roll-out a new best node
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # cut dominated nodes

                stack.append(node_new)  # add new node to stack, LIFO

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

    return node_best.t_ex, node_best.ch_ex  # optimal


def branch_bound_with_stats(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
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
    node_stats : List of TreeNode
        More nodes.
    """

    stack = [TreeNodeBound(tasks, ch_avail, rng=rng)]  # Initialize Stack
    node_stats = [TreeNodeBound(tasks, ch_avail, rng=rng)]
    # NodeStats = []

    node_best = stack[0].roll_out(inplace=False)  # roll-out initial solution
    l_best = node_best.l_ex
    node_stats.append(node_best)

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if len(node_new.seq) == len(tasks):
                # Append any complete solutions, use for training NN. Can decipher what's good/bad based on final costs
                node_stats.append(node_new)

            if node_new.l_lo < l_best:  # New node is not dominated
                if node_new.l_up < l_best:
                    node_best = node_new.roll_out(inplace=False)  # roll-out a new best node
                    # NodeStats.append(node_best)
                    l_best = node_best.l_ex
                    stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {l_best:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

    node_stats.pop(0)    # Remove First Initialization stage
    return node_best.t_ex, node_best.ch_ex, node_stats


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


def mcts(tasks, ch_avail, n_mc, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
        Channel availability times.
    n_mc : int
        Number of roll-outs performed.
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

    # TODO: add exploration/exploitation input control.

    l_up = TreeNodeBound(tasks, ch_avail).l_up
    tree = SearchNode(n_tasks=len(tasks), seq=[], l_up=l_up, rng=rng)

    node_best = None

    loss_min = float('inf')
    while tree.n_visits < n_mc:
        if verbose:
            print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')

        seq = tree.simulate()   # roll-out a complete sequence
        node = TreeNode(tasks, ch_avail, seq)    # evaluate execution times and channels, total loss

        loss = node.l_ex
        tree.backup(seq, loss)  # update search tree from leaf sequence to root

        if loss < loss_min:
            node_best, loss_min = node, loss

    return node_best.t_ex, node_best.ch_ex


def random_sequencer(tasks, ch_avail, rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
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


def earliest_release(tasks, ch_avail, do_swap=False):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
        Channel availability times.
    do_swap : bool
        Enables task swapping

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    seq = np.argsort([task.t_release for task in tasks])
    node = TreeNode(tasks, ch_avail, seq)

    if do_swap:
        node.check_swaps()

    return node.t_ex, node.ch_ex


def earliest_drop(tasks, ch_avail, do_swap=False):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : Iterable of task_scheduling.tasks.Base
    ch_avail : Iterable of float
        Channel availability times.
    do_swap : bool
        Enables task swapping.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    seq = list(np.argsort([task.t_drop for task in tasks]))
    node = TreeNode(tasks, ch_avail, seq)

    if do_swap:
        node.check_swaps()

    return node.t_ex, node.ch_ex


# def est_alg_kw(tasks, ch_avail):
#     """
#     Earliest Start Times Algorithm using FlexDAR scheduler function.
#
#     Parameters
#     ----------
#     tasks : Iterable of task_scheduling.tasks.Base
#     ch_avail : Iterable of float
#         Channel availability times.
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
#     seq = list(np.argsort([task.t_release for task in tasks]))  # Task Order
#     t_ex, ch_ex = FlexDARMultiChannelSequenceScheduler(seq, tasks, deepcopy(ch_avail), RP=100)
#
#     return t_ex, ch_ex


def ert_alg_kw(tasks, ch_avail, do_swap=False):

    seq = list(np.argsort([task.t_release for task in tasks]))
    node = TreeNode(tasks, ch_avail, seq)

    if do_swap:
        node.check_swaps()

    return node.t_ex, node.ch_ex, node.seq

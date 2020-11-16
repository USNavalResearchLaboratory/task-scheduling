"""Multi-channel Tree Search objects and algorithms."""

from time import perf_counter
from math import floor, factorial
import numpy as np

from task_scheduling.util.results import eval_loss

from task_scheduling.generators.scheduling_problems import Random as RandomProblem
from task_scheduling.tree_search import TreeNode, TreeNodeBound, SearchNode


def branch_bound(tasks: list, ch_avail: list, runtimes: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of tasks.Generic
    ch_avail : list of float
        Channel availability times.
    runtimes : list of float
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

    i_time = 0
    n_times = len(runtimes)

    stack = [TreeNodeBound(tasks, ch_avail, rng=rng)]  # Initialize Stack
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
            if perf_counter() - t_run >= runtimes[i_time]:
                yield node_best.t_ex, node_best.ch_ex
                i_time += 1
                if i_time == n_times:
                    run = False
                    break

            # if perf_counter() - t_run >= max_runtime:
            #     run = False
            #     break

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

    # return node_best.t_ex, node_best.ch_ex


def mcts(tasks: list, ch_avail: list, runtimes: list, verbose=False):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of tasks.Generic
    ch_avail : list of float
        Channel availability times.
    runtimes : list of float
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

    i_time = 0
    n_times = len(runtimes)

    l_up = TreeNodeBound(tasks, ch_avail).l_up
    tree = SearchNode(n_tasks=len(tasks), seq=[], l_up=l_up)

    node_best = None

    loss_min = float('inf')
    # while perf_counter() - t_run < max_runtime:
    while i_time < n_times:
        if verbose:
            print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')

        seq = tree.simulate()   # Roll-out a complete sequence
        node = TreeNode(tasks, ch_avail, seq)    # Evaluate execution times and channels, total loss

        loss = node.l_ex
        tree.backup(seq, loss)  # Update search tree from leaf sequence to root

        if loss < loss_min:
            node_best, loss_min = node, loss

        if perf_counter() - t_run >= runtimes[i_time]:
            yield node_best.t_ex, node_best.ch_ex
            i_time += 1

    # return node_best.t_ex, node_best.ch_ex


# def mcts_orig(tasks: list, ch_avail: list, max_runtime=float('inf'), n_mc=None, verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : list of tasks.Generic
#     ch_avail : list of float
#         Channel availability times.
#     n_mc : int or list of int
#         Number of Monte Carlo roll-outs per task.
#     max_runtime : float
#         Allotted algorithm runtime.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng
#         NumPy random number generator or seed. Default Generator if None.
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
#     t_run = perf_counter()
#     run = True
#
#     node = TreeNode(tasks, ch_avail, rng=rng)
#     node_best = node.roll_out(do_copy=True)
#
#     n_tasks = len(tasks)
#     if n_mc is None:
#         n_mc = [floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)]
#     elif type(n_mc) == int:
#         n_mc = n_tasks * [n_mc]
#
#     def _roll_outs(n_roll):
#         nonlocal node_best, run
#
#         for _ in range(n_roll):
#             node_mc = node.roll_out(do_copy=True)
#
#             if node_mc.l_ex < node_best.l_ex:  # Update best node
#                 node_best = node_mc
#
#             # Check run conditions
#             if perf_counter() - t_run >= max_runtime:
#                 run = False
#                 return
#
#     for n in range(n_tasks):
#         if verbose:
#             print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')
#
#         _roll_outs(n_mc[n])
#
#         if not run:
#             break
#
#         # Assign next task from earliest available channel
#         node.seq_extend(node_best.seq[n], check_valid=False)
#
#     return node_best.t_ex, node_best.ch_ex

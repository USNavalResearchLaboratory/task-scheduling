import copy
import numpy as np

from tree_search import TreeNode, TreeNodeBound

rng_default = np.random.default_rng()


def branch_bound(tasks: list, ch_avail: list, exhaustive=False, verbose=False, rng=rng_default):
    """Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    exhaustive : bool
        Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks  # TODO: proper style to redefine class attribute here?
    TreeNode._ch_avail_init = ch_avail
    TreeNode._exhaustive = exhaustive
    TreeNode._rng = rng

    n_ch = len(ch_avail)

    stack = [TreeNodeBound([[] for _ in range(n_ch)])]  # Initialize Stack
    l_upper_min = stack[0].l_up

    # Iterate
    while not ((len(stack) == 1) and (len(stack[0].seq_rem) == 0)):
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_upper_min:.3f}', end='\r')

        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < l_upper_min:  # New node is not dominated
                if node_new.l_up < l_upper_min:
                    l_upper_min = node_new.l_up
                    stack = [s for s in stack if s.l_lo < l_upper_min]  # Cut Dominated Nodes

                if len(node_new.seq_rem) > 0:  # Add New Node to Stack
                    stack.append(node_new)  # LIFO
                else:
                    stack.insert(0, node_new)

    if len(stack) != 1:
        raise ValueError('Multiple nodes...')  # TODO: delete

    if not all([s.l_lo == s.l_up for s in stack]):
        raise ValueError('Node bounds do not converge.')  # TODO: delete

    _check_loss(tasks, stack[0])

    t_ex, ch_ex = stack[0].t_ex, stack[0].ch_ex  # optimal

    return t_ex, ch_ex
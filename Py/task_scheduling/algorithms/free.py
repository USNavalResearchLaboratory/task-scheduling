from task_scheduling.tree_search import TreeNodeBound, TreeNode


# from sequence2schedule import FlexDARMultiChannelSequenceScheduler


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


def branch_bound_with_stats(tasks, ch_avail, verbose=False, rng=None):
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
    node_stats : List of TreeNode
        More nodes.
    """

    stack = [TreeNodeBound(tasks, ch_avail, rng=rng)]  # Initialize Stack
    # node_stats = [TreeNodeBound(tasks, ch_avail, rng=rng)]
    node_stats = []

    node_best = stack[0].roll_out(inplace=False)  # roll-out initial solution
    # node_stats.append(node_best)

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(permute=True):
            # Bound

            if len(node_new.seq) == len(tasks):
                # Append any complete solutions, use for training NN. Can decipher what's good/bad based on final costs
                node_stats.append(node_new)

            if node_new.l_lo < node_best.l_ex:  # New node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(inplace=False)  # roll-out a new best node
                    # if len(node_new.seq) == len(tasks) - 1:
                    #     node_stats.append(node_best)
                    # Don't append here needs to be a complete sequence. Line above is
                    # random draw to finish sequence, can have better solutions
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {l_best:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {node_best.l_ex:.3f}', end='\r')

    # node_stats.pop(0)    # Remove First Initialization stage
    # if len(node_stats) == 0:  # If by chance initial roll-out is best append it...
    #     node_stats.append(node_best)
    # node_stats.pop(0)

    return node_best.t_ex, node_best.ch_ex, node_stats


def mcts(tasks, ch_avail, n_mc=1, c_explore=1., verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    n_mc : int, optional
        Number of roll-outs performed.
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
    node = node.mcts(n_mc, c_explore, inplace=False, verbose=verbose, rng=rng)

    return node.t_ex, node.ch_ex


def mcts_v2(tasks, ch_avail, n_mc=1, c_explore=0., visit_threshold=0, verbose=False, rng=None):
    node = TreeNode(tasks, ch_avail, rng=rng)
    node = node.mcts_v2(n_mc, c_explore, visit_threshold, inplace=False, verbose=verbose, rng=rng)

    return node.t_ex, node.ch_ex


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


def earliest_release(tasks, ch_avail, check_swaps=False):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    check_swaps : bool
        Enables task swapping

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail)
    node.earliest_release(check_swaps=check_swaps)

    return node.t_ex, node.ch_ex


def earliest_drop(tasks, ch_avail, check_swaps=False):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    check_swaps : bool
        Enables task swapping.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNode(tasks, ch_avail)
    node.earliest_drop(check_swaps=check_swaps)

    return node.t_ex, node.ch_ex

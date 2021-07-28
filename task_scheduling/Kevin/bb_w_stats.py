import numpy as np

from task_scheduling.tree_search import TreeNode, TreeNodeBound
from task_scheduling.generators import scheduling_problems as problem_gens

np.set_printoptions(precision=3)


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


def _disprove():
    """Assess whether or not `branch_bound_with_stats` concept is valid."""

    # seed = None
    seed = 12348

    rng = np.random.default_rng(seed)

    n_tasks = 8
    n_ch = 1

    problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=rng)

    n_mc = 1000
    for i_mc, (tasks, ch_avail) in enumerate(problem_gen(n_mc)):
        print(f"{i_mc+1}/{n_mc}", end='\r')

        t_ex, __, node_stats = branch_bound_with_stats(tasks, ch_avail, rng=rng)
        node_opt = TreeNode(tasks, ch_avail, seq=np.argsort(t_ex))

        # FIXME: optimal node not guaranteed to be in `node_stats`
        # assert node_opt in node_stats
        if node_opt not in node_stats:
            node_stats.append(node_opt)

        nodes = np.array([(node.seq, node.l_ex, -1) for node in node_stats],
                         dtype=[('seq', int, (n_tasks,)), ('l_ex', float), ('start', int)])

        # FIXME: elements of `node_stats` are not unique
        __, idx_unique = np.unique(nodes['seq'], axis=0, return_index=True)
        nodes = nodes[idx_unique]
        node_stats = [node_stats[i] for i in idx_unique]

        # Determine partial sequences for each node
        nodes['start'][node_stats.index(node_opt)] = 0      # optimal solution start index
        for n in range(1, n_tasks):
            idx = np.flatnonzero(nodes['start'] == -1)      # currently undetermined nodes

            seqs = nodes['seq'][:, :n]
            for seq in np.unique(seqs, axis=0):
                i = np.flatnonzero(np.all(seqs == seq, axis=1))     # matching partial sequences
                i_min = i[np.argmin(nodes['l_ex'][i])]              # lowest loss of candidates
                if i_min in idx:
                    nodes['start'][i_min] = n       # if undetermined, set the start index

        # Assess partial solutions
        for node in nodes:
            seq_partial = node['seq'][:node['start']]
            node_solve = TreeNodeBound(tasks, ch_avail, seq=seq_partial, rng=None).branch_bound(inplace=False)

            assert np.isclose(node['l_ex'], node_solve.l_ex)

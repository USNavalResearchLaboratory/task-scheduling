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


#%% MCTS restricted by search count

# def mcts(self, n_mc=1, c_explore=0., visit_threshold=0, inplace=True, verbose=False, rng=None):
#     """
#     Monte Carlo tree search.
#
#     Parameters
#     ----------
#     n_mc : int, optional
#         Number of complete sequences evaluated.
#     c_explore : float, optional
#         Exploration weight. Higher values prioritize less frequently visited notes.
#     visit_threshold : int, optional
#         Nodes with up to this number of visits will select children using the `expansion` method.
#     inplace : bool, optional
#         If True, self.seq is completed. Otherwise, a new node object is returned.
#     verbose : bool, optional
#         Enables printing of algorithm state information.
#     rng : int or RandomState or Generator, optional
#         NumPy random number generator or seed. Instance RNG if None.
#
#     Returns
#     -------
#     TreeNode, optional
#         Only if `inplace` is False.
#
#     """
#
#     rng = self._get_rng(rng)
#
#     bounds = TreeNodeBound(self.tasks, self.ch_avail).bounds
#     root = SearchNode(self.n_tasks, bounds, self.seq, c_explore, visit_threshold, rng=rng)
#
#     node_best, loss_best = None, np.inf
#     while root.n_visits < n_mc:
#         if verbose:
#             print(f'Solutions evaluated: {root.n_visits+1}/{n_mc}, Min. Loss: {loss_best}', end='\r')
#
#         leaf_new = root.selection()  # expansion step happens in selection call
#
#         seq_ext = leaf_new.seq[len(self.seq):]
#         node = self._extend_util(seq_ext, inplace=False)
#         node.roll_out()
#         if node.l_ex < loss_best:
#             node_best, loss_best = node, node.l_ex
#
#         # loss = leaf_new.evaluation()
#         loss = node.l_ex
#         leaf_new.backup(loss)
#
#     if inplace:
#         # self.seq = node_best.seq
#         seq_ext = node_best.seq[len(self.seq):]
#         self._extend_util(seq_ext)
#     else:
#         return node_best
#
# def mcts_v1(self, n_mc=1, c_explore=1., inplace=True, verbose=False, rng=None):
#
#     rng = self._get_rng(rng)
#
#     tree = SearchNodeV1(self.n_tasks, self.seq, c_explore=c_explore, rng=rng)
#
#     node_best, loss_best = None, np.inf
#     while tree.n_visits < n_mc:
#         if verbose:
#             print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_best}', end='\r')
#
#         seq = tree.simulate()  # roll-out a complete sequence
#
#         seq_ext = seq[len(self.seq):]
#         node = self._extend_util(seq_ext, inplace=False)
#         if node.l_ex < loss_best:
#             node_best, loss_best = node, node.l_ex
#
#         tree.backup(seq, node.l_ex)  # update search tree from leaf sequence to root
#
#     if inplace:
#         # self.seq = node_best.seq
#         seq_ext = node_best.seq[len(self.seq):]
#         self._extend_util(seq_ext)
#     else:
#         return node_best

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

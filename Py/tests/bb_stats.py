from collections import deque

import numpy as np

from task_scheduling.tree_search import TreeNode, TreeNodeBound
from task_scheduling.algorithms.free import branch_bound_with_stats
from task_scheduling.generators import scheduling_problems as problem_gens, tasks as task_gens
from task_scheduling.util.results import eval_loss

np.set_printoptions(precision=3)


# seed = None
seed = 12345

rng = np.random.default_rng(seed)

n_tasks = 4
n_ch = 1

problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=rng)

n_mc = 1000
cnt = 0
for i_mc, (tasks, ch_avail) in enumerate(problem_gen(n_mc)):
    print(f"{i_mc+1}/{n_mc}", end='\r')

    t_ex, __, node_stats = branch_bound_with_stats(tasks, ch_avail, rng=rng)
    node_opt = TreeNode(tasks, ch_avail, seq=np.argsort(t_ex))

    # assert node_opt in node_stats       # FIXME: optimal node not guaranteed to be in `node_stats`
    if node_opt not in node_stats:
        node_stats.append(node_opt)

    # FIXME: elements of `node_stats` are not unique
    nodes = np.array([(node.seq, node.l_ex, -1) for node in node_stats],
                     dtype=[('seq', int, (n_tasks,)), ('l_ex', float), ('start', int)])
    __, idx_unique = np.unique(nodes['seq'], axis=0, return_index=True)
    nodes = nodes[idx_unique]
    node_stats = [node_stats[i] for i in idx_unique]


    nodes['start'][node_stats.index(node_opt)] = 0
    idx = 1
    while np.any(nodes['start'] == -1):
        temp = nodes['seq'][:, :idx]
        for sp in np.unique(temp, axis=0):
            idx = np.flatnonzero(np.all(temp == sp, axis=1))

        idx += 1


    idx = 1
    while len(node_stats) > 0:
        nodes = np.array([(node.seq, node.l_ex) for node in node_stats],
                         dtype=[('seq', int, (n_tasks,)), ('l_ex', float)])

        temp = nodes['seq'][:, :idx]
        for sp in np.unique(temp, axis=0):
            idx = np.flatnonzero(np.all(temp == sp, axis=1))

        idx += 1

    # nodes = np.array([(node.seq, node.l_ex) for node in node_stats], dtype=[('seq', int, (n_tasks,)), ('l_ex', float)])
    # nodes = np.unique(nodes, axis=0)
    #
    # cnt += len(nodes)
    #



    # for seq, l_ex in nodes:
    #     n_0 = seq[0]
    #     if np.sum(seqs[:, 0] == n_0) == 1:
    #         _node = TreeNodeBound(tasks, ch_avail, seq=[n_0], rng=None).branch_bound(inplace=False)
    #         seq_opt_sub = _node.seq
    #         l_opt_sub = _node.l_ex
    #
    #         # print(TreeNode(tasks, ch_avail, seq=seq).l_ex)
    #         # print(TreeNode(tasks, ch_avail, seq=seq_opt_sub).l_ex)
    #
    #         # if not np.isclose(TreeNode(tasks, ch_avail, seq=seq).l_ex, TreeNode(tasks, ch_avail, seq=seq_opt_sub).l_ex):
    #         #     t_ex, ch_ex, node_stats = branch_bound_with_stats(tasks, ch_avail, rng=seed)
    #
    #         assert np.isclose(TreeNode(tasks, ch_avail, seq=seq).l_ex, l_opt_sub)
    #         # assert seq == seq_opt_sub


    # seqs = np.array([node.seq for node in node_stats])
    # seqs = np.unique(seqs, axis=0)

    # assert seq_opt in seqs

    # if len(node_stats) == 0:  # Should never have empty node_stats? Re-run to investigate
    #     t_ex, ch_ex, node_stats = branch_bound_with_stats(tasks, ch_avail, rng=seed)

    # if not np.all(seq_opt == seqs[-1, :]):
    #     a = 1

    # if seqs.ndim == 1:
    #     seqs = seq_opt[np.newaxis]
    # elif seq_opt not in seqs:
    #     seqs = np.concatenate((seqs, seq_opt), axis=0)
    #     # seqs.append(seq_opt)

    # for seq in seqs:
    #     n_0 = seq[0]
    #     if np.sum(seqs[:, 0] == n_0) == 1:
    #         _node = TreeNodeBound(tasks, ch_avail, seq=[n_0], rng=None).branch_bound(inplace=False)
    #         seq_opt_sub = _node.seq
    #         l_opt_sub = _node.l_ex
    #
    #         # print(TreeNode(tasks, ch_avail, seq=seq).l_ex)
    #         # print(TreeNode(tasks, ch_avail, seq=seq_opt_sub).l_ex)
    #
    #         # if not np.isclose(TreeNode(tasks, ch_avail, seq=seq).l_ex, TreeNode(tasks, ch_avail, seq=seq_opt_sub).l_ex):
    #         #     t_ex, ch_ex, node_stats = branch_bound_with_stats(tasks, ch_avail, rng=seed)
    #
    #         assert np.isclose(TreeNode(tasks, ch_avail, seq=seq).l_ex, l_opt_sub)
    #         # assert seq == seq_opt_sub

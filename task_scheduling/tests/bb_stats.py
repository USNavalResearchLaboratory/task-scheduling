"""
Assess whether or not `branch_bound_with_stats` concept is valid.
"""
import numpy as np

from task_scheduling.tree_search import TreeNode, TreeNodeBound
from task_scheduling.algorithms.free import branch_bound_with_stats
from task_scheduling.generators import scheduling_problems as problem_gens, tasks as task_gens

np.set_printoptions(precision=3)


# seed = None
seed = 12348

rng = np.random.default_rng(seed)

n_tasks = 8
n_ch = 1

problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=rng)

n_mc = 1000
cnt = 0
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

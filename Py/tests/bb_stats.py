from collections import deque

import numpy as np

from task_scheduling.tree_search import TreeNode, TreeNodeBound
from task_scheduling.algorithms.free import branch_bound, branch_bound_with_stats
from task_scheduling.generators import scheduling_problems as problem_gens, tasks as task_gens
from task_scheduling.util.results import eval_loss


def bb_from_node(node):     # FIXME: implement as Node method!?

    stack = deque([node])        # initialize stack
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

    return node_best


n_ch = 1
n_tasks = 4

n_mc = 1
for i_mc in range(n_mc):
    print(f"{i_mc+1}/{n_mc}")

    problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks, n_ch, ch_avail_lim=(0., 0.))

    (tasks, ch_avail), = problem_gen(1)

    t_ex, ch_ex, node_stats = branch_bound_with_stats(tasks, ch_avail)
    seq_opt = np.argsort(t_ex)
    l_opt = eval_loss(tasks, t_ex)

    # nodes = np.array([(node.seq, node.l_ex) for node in node_stats], dtype=[('seq', int, (n_tasks,)), ('l_ex', float)])
    seqs = np.array([node.seq for node in node_stats])
    if seq_opt not in seqs:
        seqs = np.concatenate((seqs, seq_opt), axis=0)

    for seq in seqs.tolist():
        n_0 = seq[0]
        if (seqs[:, 0] == n_0).sum() == 1:
            _node = TreeNodeBound(tasks, ch_avail, seq=[n_0])
            seq_opt_sub = bb_from_node(_node).seq

            assert seq == seq_opt_sub

"""
Branch and Bound.
"""

import numpy as np
rng = np.random.default_rng()


class TreeNode:
    _tasks = None       # TODO: needs to be overwritten by invoking scripts...

    def __init__(self, seq, t_ex, l_inc, lb, ub):
        if TreeNode._tasks is None:
            raise AttributeError("Cannot instantiate objects before assigning the '_tasks' class attribute")

        self.seq = seq          # partial task index sequence
        self.t_ex = t_ex        # task execution times
        self.l_inc = l_inc      # partial incurred loss
        self.lb = lb            # loss lower bound
        self.ub = ub            # loss upper bound

    # TODO: make everything but t_ex read-only properties, updated on setter!

    @property
    def t_avail(self):      # TODO: use a setter and protected attribute to minimize computation?
        if len(self.seq) == 0:
            return 0
        else:
            return self.t_ex[self.seq[-1]] + self._tasks[self.seq[-1]].duration

    def branch(self):
        """Generate New Nodes"""

        nodes_new = []

        tasks = TreeNode._tasks

        N = len(tasks)
        seq_c = set(range(N)) - set(self.seq)
        for n in seq_c:
            seq = self.seq + [n]        # append to new sequence

            t_ex = self.t_ex.copy()           # formulate execution time for new task   TODO: need copy()?
            t_ex[n] = max(tasks[n].t_start, self.t_avail)

            l_inc = self.l_inc + tasks[n].loss_fcn(t_ex[n])     # update partial loss

            seq_c = set(range(N)) - set(seq)       # set of remaining tasks

            t_end = t_ex[n] + tasks[n].duration
            t_s_max = max([tasks[i].t_start for i in seq_c] + [t_end]) + sum([tasks[i].duration for i in seq_c])

            lb = l_inc
            ub = l_inc
            for i in seq_c:       # update loss bounds
                lb += tasks[i].loss_fcn(max(tasks[i].t_start, t_end))
                ub += tasks[i].loss_fcn(t_s_max)

            nodes_new.append(TreeNode(seq, t_ex, l_inc, lb, ub))

        return nodes_new


def branch_bound(tasks, verbose=False):
    """Branch and Bound algorithm."""

    # Initialize Stack
    N = len(tasks)

    TreeNode._tasks = tasks         # TODO: proper style to redefine class attribute here?

    S = [TreeNode(seq=[], t_ex=np.full(N, np.nan), l_inc=0., lb=0., ub=np.inf)]

    # Iterate
    while not ((len(S) == 1) and (len(S[0].seq) == N)):
        if verbose:
            print(f'# Remaining Branches = {len(S)}', end='\n')

        node = S.pop()     # Extract Node

        # Branch
        for node_new in rng.permutation(node.branch()):
            # Bound
            if node_new.lb < min([b.ub for b in S] + [np.inf]):        # New node is not dominated
                S = [b for b in S if b.lb < node_new.ub]      # Cut Dominated Nodes

                if len(node_new.seq) < N:  # Add New Node to Stack
                    S.append(node_new)     # LIFO
                else:
                    S.insert(0, node_new)

    if len(S) != 1:
        raise ValueError('Multiple nodes...')

    if not all([b.lb == b.ub for b in S]):
        raise ValueError('Node bounds do not converge.')

    t_ex_opt = S[0].t_ex
    l_opt = S[0].l_inc

    return t_ex_opt, l_opt



"""
Branch and Bound.
"""

import copy

import numpy as np
rng = np.random.default_rng()


class TreeNode:
    _tasks = None       # TODO: needs to be overwritten by invoking scripts...
    _do_bounds = None

    def __init__(self, seq):
        if TreeNode._tasks is None or TreeNode._do_bounds is None:
            raise AttributeError("Cannot instantiate objects before assigning the '_tasks' and '_do_bounds class attributes")

        self._seq = []      # partial task index sequence
        self._t_ex = np.full(len(self._tasks), np.nan)      # task execution times (NaN for unscheduled)

        self._t_avail = 0.  # timeline availability

        self._l_inc = 0.    # partial sequence loss
        self._l_lo = None
        self._l_up = None

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(seq={self.seq})"

    @property
    def seq(self): return self._seq

    @property
    def t_ex(self): return self._t_ex

    @property
    def t_avail(self): return self._t_avail

    @property
    def l_inc(self): return self._l_inc

    @property
    def l_lo(self): return self._l_lo       # TODO: suppress if not do_bounds?

    @property
    def l_up(self): return self._l_up


    @seq.setter
    def seq(self, seq):
        """Node Update. Sets 'seq' and iteratively updates all attributes."""

        if self._seq != seq[:len(self._seq)]:       # new sequence is not an extension of current sequence
            self.__init__(seq)  # initialize from scratch

        seq_append = seq[len(self._seq):]       # new task indices to schedule

        self._seq = seq

        for n in seq_append:        # recursively update Node attributes
            self._t_ex[n] = max(self._tasks[n].t_start, self._t_avail)
            self._t_avail = self._t_ex[n] + self._tasks[n].duration
            self._l_inc += self._tasks[n].loss_fcn(self._t_ex[n])

        if TreeNode._do_bounds:
            seq_rem = set(range(len(self._tasks))) - set(seq)       # remaining unscheduled task indices
            t_ex_max = max([self._tasks[n].t_start for n in seq_rem] + [self._t_avail]) \
                       + sum([self._tasks[n].duration for n in seq_rem])        # maximum execution time for bounding

            self._l_lo = self._l_inc
            self._l_up = self._l_inc
            for n in seq_rem:     # update loss bounds
                self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_start, self._t_avail))
                self._l_up += self._tasks[n].loss_fcn(t_ex_max)


    def branch(self):
        """Generate All Sub-Nodes."""

        seq_rem = set(range(len(self._tasks))) - set(self.seq)

        nodes_new = []
        for n in seq_rem:
            node_new = copy.deepcopy(self)          # new Node object
            node_new.seq = node_new.seq + [n]       # invoke seq.setter method
            nodes_new.append(node_new)

        return nodes_new


    def roll_out(self, do_copy=True):
        seq_rem = set(range(len(self._tasks))) - set(self.seq)      # TODO: make this a private attribute?

        seq_rem_perm = list(rng.permutation(list(seq_rem)))
        if do_copy:
            node_new = copy.deepcopy(self)      # new Node object
            node_new.seq = node_new.seq + seq_rem_perm      # invoke seq.setter method

            return node_new
        else:
            self.seq = self.seq + seq_rem_perm





def branch_bound(tasks, verbose=False):
    """Branch and Bound algorithm."""

    TreeNode._tasks = tasks         # TODO: proper style to redefine class attribute here?
    TreeNode._do_bounds = True

    S = [TreeNode([])]      # Initialize Stack

    # TODO: slow? keep track of bounds?

    # Iterate
    while not ((len(S) == 1) and (len(S[0].seq) == len(tasks))):
        if verbose:
            print(f'# Remaining Nodes = {len(S)}', end='\n')

        node = S.pop()     # Extract Node

        # Branch
        for node_new in rng.permutation(node.branch()):
            # Bound
            if node_new.l_lo < min([b.l_up for b in S] + [np.inf]):        # New node is not dominated
                S = [b for b in S if b.l_lo < node_new.l_up]      # Cut Dominated Nodes

                if len(node_new.seq) < len(tasks):  # Add New Node to Stack
                    S.append(node_new)     # LIFO
                else:
                    S.insert(0, node_new)

    # if len(S) != 1:
    #     raise ValueError('Multiple nodes...')
    #
    # if not all([b.l_lo == b.l_up for b in S]):
    #     raise ValueError('Node bounds do not converge.')

    t_ex_opt = S[0].t_ex
    l_opt = S[0].l_inc

    return t_ex_opt, l_opt


def mc_tree_search(tasks, N_mc, verbose=False):
    TreeNode._tasks = tasks
    TreeNode._do_bounds = False

    node = TreeNode([])
    node_mc_best = node.roll_out(do_copy=True)

    N = len(tasks)
    for n in range(N):
        if verbose:
            print(f'Assigning Task {n+1}/{N}')

        # Perform Rollouts
        for _ in range(N_mc):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_inc < node_mc_best.l_inc:   # Update best node
                node_mc_best = node_mc     # TODO: copy?

        node.seq = node.seq + [node_mc_best.seq[n]]

    t_ex_opt = node.t_ex
    l_opt = node.l_inc

    return t_ex_opt, l_opt




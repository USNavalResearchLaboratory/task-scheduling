"""
Branch and Bound.
"""

import copy

import numpy as np

rng_default = np.random.default_rng()


class TreeNode:
    _tasks = None       # TODO: needs to be overwritten by invoking scripts...
    rng = None

    def __init__(self, seq):
        if TreeNode._tasks is None:
            raise AttributeError("Cannot instantiate objects before assigning the '_tasks' and '_do_bounds class attributes")

        self._seq = []      # partial task index sequence
        self._t_ex = np.full(len(self._tasks), np.nan)      # task execution times (NaN for unscheduled)

        self._t_avail = 0.  # timeline availability

        self._l_ex = 0.    # partial sequence loss

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, partial loss:{self.l_ex:.3f})"

    @property
    def seq(self): return self._seq

    @seq.setter
    def seq(self, seq):
        self.update_node(seq)

    @property
    def t_ex(self): return self._t_ex

    @property
    def t_avail(self): return self._t_avail

    @property
    def l_ex(self): return self._l_ex

    def update_node(self, seq):
        """Node Update. Sets 'seq' and iteratively updates all attributes."""

        if self._seq != seq[:len(self._seq)]:  # new sequence is not an extension of current sequence
            self.__init__(seq)  # initialize from scratch

        seq_append = seq[len(self._seq):]  # new task indices to schedule

        self._seq = seq

        for n in seq_append:  # recursively update Node attributes
            self._t_ex[n] = max(self._tasks[n].t_start, self._t_avail)
            self._t_avail = self._t_ex[n] + self._tasks[n].duration
            self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True):
        """Generate All Sub-Nodes."""

        seq_rem = set(range(len(self._tasks))) - set(self.seq)

        nodes_new = []
        for n in seq_rem:
            node_new = copy.deepcopy(self)          # new Node object
            node_new.seq = node_new.seq + [n]       # invoke seq.setter method
            nodes_new.append(node_new)

        if do_permute:
            nodes_new = self.rng.permutation(nodes_new)

        return nodes_new

    def roll_out(self, do_copy=True):
        seq_rem = set(range(len(self._tasks))) - set(self.seq)      # TODO: make this a private attribute?

        seq_rem_perm = list(self.rng.permutation(list(seq_rem)))
        if do_copy:
            node_new = copy.deepcopy(self)      # new Node object
            node_new.seq = node_new.seq + seq_rem_perm      # invoke seq.setter method

            return node_new
        else:
            self.seq = self.seq + seq_rem_perm


class TreeNodeBound(TreeNode):
    def __init__(self, seq):
        self._l_lo = 0.
        self._l_up = np.inf
        super().__init__(seq)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo}:.3f < loss < {self.l_up}:.3f)"

    @property
    def l_lo(self): return self._l_lo

    @property
    def l_up(self): return self._l_up

    @TreeNode.seq.setter
    def seq(self, seq):
        self.update_node(seq)

        # Add bound attributes
        seq_rem = set(range(len(self._tasks))) - set(seq)  # remaining unscheduled task indices
        t_ex_max = max([self._tasks[n].t_start for n in seq_rem] + [self._t_avail]) \
                   + sum([self._tasks[n].duration for n in seq_rem])  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_start, self._t_avail))
            self._l_up += self._tasks[n].loss_fcn(t_ex_max)


def branch_bound(tasks, verbose=False, rng=rng_default):
    """Branch and Bound algorithm."""

    TreeNode._tasks = tasks         # TODO: proper style to redefine class attribute here?
    TreeNode.rng = rng

    stack = [TreeNodeBound([])]      # Initialize Stack

    l_upper_min = stack[0].l_up

    # Iterate
    while not ((len(stack) == 1) and (len(stack[0].seq) == len(tasks))):
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}', end='\n')

        node = stack.pop()     # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < l_upper_min:  # New node is not dominated
                stack = [s for s in stack if s.l_lo < node_new.l_up]      # Cut Dominated Nodes
                l_upper_min = min(l_upper_min, node_new.l_up)

                if len(node_new.seq) < len(tasks):  # Add New Node to Stack
                    stack.append(node_new)     # LIFO
                else:
                    stack.insert(0, node_new)

    if len(stack) != 1:
        raise ValueError('Multiple nodes...')

    if not all([s.l_lo == s.l_up for s in stack]):
        raise ValueError('Node bounds do not converge.')

    t_ex_opt = stack[0].t_ex
    l_opt = stack[0].l_ex

    return t_ex_opt, l_opt


def mc_tree_search(tasks, N_mc, verbose=False, rng=rng_default):
    TreeNode._tasks = tasks
    TreeNode.rng = rng

    node = TreeNode([])
    node_mc_best = node.roll_out(do_copy=True)

    N = len(tasks)
    for n in range(N):
        if verbose:
            print(f'Assigning Task {n+1}/{N}')

        # Perform Roll-outs
        for _ in range(N_mc):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_mc_best.l_ex:   # Update best node
                node_mc_best = node_mc

        node.seq = node.seq + [node_mc_best.seq[n]]     # invokes seq.setter

    t_ex_opt = node.t_ex
    l_opt = node.l_ex

    return t_ex_opt, l_opt




"""
Branch and Bound.
"""

import copy

import numpy as np

rng_default = np.random.default_rng()


def _check_loss(tasks, node):
    l_ex = node.l_ex
    l_eval = 0
    for n in range(len(tasks)):
        l_eval += tasks[n].loss_fcn(node.t_ex[n])
    if abs(l_eval - l_ex) > 1e-12:
        raise ValueError('Iterated loss is inaccurate')


class TreeNode:
    _tasks = []       # TODO: needs to be overwritten by invoking scripts...
    rng = None

    def __init__(self, seq):
        if self._n_tasks == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' class attribute.")

        self._seq = []      # partial task index sequence
        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)      # task execution times (NaN for unscheduled)

        self._t_avail = 0.  # timeline availability

        self._l_ex = 0.    # partial sequence loss

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, partial loss:{self.l_ex:.3f})"

    @property
    def _n_tasks(self): return len(self._tasks)

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
        """Node Update. Sets 'seq' and iteratively updates all dependent attributes."""

        if np.array(seq).size != np.unique(seq).size:
            raise ValueError("Input 'seq' must have unique values.")

        if self._seq != seq[:len(self._seq)]:   # new sequence is not an extension of current sequence
            self.__init__(seq)                  # initialize from scratch

        seq_append = seq[len(self._seq):]  # new task indices to schedule
        self._seq_rem = self._seq_rem - set(seq_append)

        self._seq = seq

        for n in seq_append:  # recursively update Node attributes
            self._t_ex[n] = max(self._tasks[n].t_release, self._t_avail)
            self._t_avail = self._t_ex[n] + self._tasks[n].duration
            self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True):
        """Generate All Sub-Nodes."""

        nodes_new = []
        for n in self._seq_rem:
            seq_new = copy.deepcopy(self.seq)
            seq_new.append(n)

            node_new = copy.deepcopy(self)  # new Node object
            node_new.seq = seq_new          # invoke seq.setter method
            # node_new.seq = node_new.seq + [n]       # invoke seq.setter method

            nodes_new.append(node_new)

        if do_permute:
            nodes_new = self.rng.permutation(nodes_new)

        return nodes_new

    def roll_out(self, do_copy=True):
        """Roll-out remaining sequence randomly."""

        seq_new = copy.deepcopy(self.seq)
        seq_rem_perm = self.rng.permutation(list(self._seq_rem)).tolist()
        seq_new.extend(seq_rem_perm)

        if do_copy:
            node_new = copy.deepcopy(self)      # new Node object
            node_new.seq = seq_new  # invoke seq.setter method
            # node_new.seq = node_new.seq + seq_rem_perm      # invoke seq.setter method

            return node_new
        else:
            self.seq = seq_new  # invoke seq.setter method
            # self.seq = self.seq + seq_rem_perm


class TreeNodeBound(TreeNode):
    def __init__(self, seq):
        self._l_lo = 0.
        self._l_up = np.inf
        super().__init__(seq)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    @property
    def l_lo(self): return self._l_lo

    @property
    def l_up(self): return self._l_up

    @TreeNode.seq.setter        # TODO: better way to overwrite setter method?
    def seq(self, seq):
        self.update_node(seq)

        # Add bound attributes
        t_ex_max = max([self._tasks[n].t_release for n in self._seq_rem] + [self._t_avail]) \
            + sum([self._tasks[n].duration for n in self._seq_rem])  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_release, self._t_avail))
            self._l_up += self._tasks[n].loss_fcn(t_ex_max)


def branch_bound(tasks, verbose=False, rng=rng_default):
    """Branch and Bound algorithm."""

    TreeNode._tasks = tasks         # TODO: proper style to redefine class attribute here?
    TreeNode.rng = rng

    n_tasks = len(tasks)

    stack = [TreeNodeBound([])]      # Initialize Stack
    l_upper_min = stack[0].l_up

    # Iterate
    while not ((len(stack) == 1) and (len(stack[0].seq) == n_tasks)):
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}', end='\n')

        node = stack.pop()     # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < l_upper_min:  # New node is not dominated
                stack = [s for s in stack if s.l_lo < node_new.l_up]      # Cut Dominated Nodes
                l_upper_min = min(l_upper_min, node_new.l_up)

                if len(node_new.seq) < n_tasks:  # Add New Node to Stack
                    stack.append(node_new)     # LIFO
                else:
                    stack.insert(0, node_new)

    if len(stack) != 1:
        raise ValueError('Multiple nodes...')

    if not all([s.l_lo == s.l_up for s in stack]):
        raise ValueError('Node bounds do not converge.')

    _check_loss(tasks, stack[0])

    t_ex = stack[0].t_ex        # optimal

    return t_ex


def mc_tree_search(tasks, n_mc, verbose=False, rng=rng_default):
    TreeNode._tasks = tasks
    TreeNode.rng = rng

    node = TreeNode([])
    node_mc_best = node.roll_out(do_copy=True)

    n_tasks = len(tasks)
    for n in range(n_tasks):
        if verbose:
            print(f'Assigning Task {n+1}/{n_tasks}')

        # Perform Roll-outs
        for _ in range(n_mc):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_mc_best.l_ex:   # Update best node
                node_mc_best = node_mc

        node.seq = node.seq + [node_mc_best.seq[n]]     # invokes seq.setter

    _check_loss(tasks, node)

    t_ex = node.t_ex

    return t_ex

"""Multi-channel Tree Search objects and algorithms."""

import copy
import numpy as np

rng_default = np.random.default_rng()


class TreeNode:
    """Node object for tree search algorithms.

    Parameters
    ----------
    seq : list of list
        List of task index sequences by channel

    Attributes
    ----------
    seq : list of list
        Partial task index sequence.
    t_ex : ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : ndarray
        Task execution channels. NaN for unscheduled.
    t_avail : ndarray
        Channel availability times.
    l_ex : float
        Total loss of scheduled tasks.
    seq_rem: set
        Unscheduled task indices.

    """
    # TODO: docstring describes properties as attributes. OK? Subclasses, too.

    _tasks = []       # TODO: needs to be overwritten by invoking scripts...
    _n_ch = 0
    _rng = None

    # TODO: variable channel availability initialization

    def __init__(self, seq: list):
        if self._n_tasks == 0 or TreeNode._n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and '_n_ch class attributes.")

        self._seq = [[] for _ in range(self._n_ch)]

        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)      # task execution times (NaN for unscheduled)
        self._ch_ex = np.full(self._n_tasks, np.nan, dtype=np.int)

        self._t_avail = np.zeros(self._n_ch)    # timeline availability

        self._l_ex = 0.    # partial sequence loss

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, partial loss:{self.l_ex:.3f})"

    @property
    def _n_tasks(self): return len(self._tasks)

    @property
    def seq(self):
        """Gets the node sequence. Setter calls 'update_node'.

        Returns
        -------
        list of list
            List of task index sequences by channel

        """
        return self._seq

    @seq.setter
    def seq(self, seq):
        self.update_node(seq)

    @property
    def t_ex(self): return self._t_ex

    @property
    def ch_ex(self): return self._ch_ex

    @property
    def t_avail(self): return self._t_avail

    @property
    def l_ex(self): return self._l_ex

    @property
    def seq_rem(self): return self._seq_rem

    def update_node(self, seq: list):
        """Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.

        """

        seq_cat = np.concatenate(seq)
        if seq_cat.size != np.unique(seq_cat).size:
            raise ValueError("Input 'seq' must have unique values.")

        for ch in range(self._n_ch):
            if self._seq[ch] != seq[ch][:len(self._seq[ch])]:   # new sequence is not an extension of current sequence
                self.__init__(seq)                              # initialize from scratch
                break

        for ch in range(self._n_ch):
            seq_append = seq[ch][len(self._seq[ch]):]   # new task indices to schedule
            self._seq_rem = self._seq_rem - set(seq_append)

            self._seq[ch] = seq[ch]

            self._ch_ex[seq_append] = ch
            for n in seq_append:    # recursively update Node attributes
                self._t_ex[n] = max(self._tasks[n].t_release, self._t_avail[ch])
                self._t_avail[ch] = self._t_ex[n] + self._tasks[n].duration
                self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True, exhaustive=False):
        """Generate descendant nodes.

        Parameters
        ----------
        do_permute : bool
            Enables random permutation of returned node list.
        exhaustive : bool
            Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.

        Yields
        -------
        TreeNode
            Descendant node with one additional task scheduled.

        """

        seq_iter = self._seq_rem
        if do_permute:
            seq_iter = self._rng.permutation(list(seq_iter))

        for n in seq_iter:

            if exhaustive:
                ch_iter = range(self._n_ch)
                if do_permute:
                    ch_iter = self._rng.permutation(list(ch_iter))  # try each task on each channel
            else:
                ch_iter = [int(np.argmin(self.t_avail))]  # try each task on the earliest available channel only

            for ch in ch_iter:
                seq_new = copy.deepcopy(self.seq)
                seq_new[ch].append(n)

                node_new = copy.deepcopy(self)  # new Node object
                node_new.seq = seq_new  # call seq.setter method
                # node_new.seq = node_new.seq + [n]       # call seq.setter method

                yield node_new

    def roll_out(self, do_copy=True):
        """Generates/updates node with a randomly completed sequence.

        Parameters
        ----------
        do_copy : bool
            Enables return of a new TreeNode object. Otherwise, updates in-place.

        Returns
        -------
        TreeNode
            Only if do_copy is True.

        """

        seq_new = copy.deepcopy(self.seq)

        seq_rem_perm = self._rng.permutation(list(self._seq_rem))

        temp = self._rng.multinomial(self._n_tasks, np.ones(self._n_ch) / self._n_ch)
        i_split = np.cumsum(temp)[:-1]
        splits = np.split(seq_rem_perm, i_split)
        for ch in range(self._n_ch):
            seq_new[ch].extend(splits[ch].tolist())

        if do_copy:
            node_new = copy.deepcopy(self)      # new Node object
            node_new.seq = seq_new  # call seq.setter method

            return node_new
        else:
            self.seq = seq_new  # call seq.setter method


class TreeNodeBound(TreeNode):
    """Node object with additional loss bounding attributes.

        Parameters
        ----------
        seq : list of list
            List of task index sequences by channel

        Attributes
        ----------
        seq : list of list
            Partial task index sequence.
        t_ex : ndarray
            Task execution times. NaN for unscheduled.
        ch_ex : ndarray
            Task execution channels. NaN for unscheduled.
        t_avail : ndarray
            Channel availability times.
        l_ex : float
            Total loss of scheduled tasks.
        seq_rem: set
            Unscheduled task indices.
        l_lo: float
            Lower bound on total loss for descendant nodes.
        l_up: float
            Upper bound on total loss for descendant nodes.

        """

    def __init__(self, seq: list):
        self._l_lo = 0.
        self._l_up = float('inf')
        super().__init__(seq)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    @property
    def l_lo(self): return self._l_lo

    @property
    def l_up(self): return self._l_up

    # @TreeNode.seq.setter
    # def seq(self, seq):
    #     self.update_node(seq)
    #
    #     # Add bound attributes
    #     t_ex_max = max([self._tasks[n].t_release for n in self._seq_rem] + list(self._t_avail)) \
    #         + sum([self._tasks[n].duration for n in self._seq_rem])  # maximum execution time for bounding
    #
    #     self._l_lo = self._l_ex
    #     self._l_up = self._l_ex
    #     for n in self._seq_rem:  # update loss bounds
    #         self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_release, min(self._t_avail)))
    #         self._l_up += self._tasks[n].loss_fcn(t_ex_max)
    #
    #     if len(self._seq_rem) > 0 and self._l_lo == self._l_up:     # roll-out if bounds converge
    #         self.roll_out(do_copy=False)

    def update_node(self, seq: list):
        """Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.

        """

        super().update_node(seq)

        # Add bound attributes
        t_ex_max = (max([self._tasks[n].t_release for n in self._seq_rem] + list(self._t_avail))
                    + sum([self._tasks[n].duration for n in self._seq_rem]))  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_release, min(self._t_avail)))
            self._l_up += self._tasks[n].loss_fcn(t_ex_max)

        if len(self._seq_rem) > 0 and self._l_lo == self._l_up:  # roll-out if bounds converge
            self.roll_out(do_copy=False)


def _check_loss(tasks: list, node: TreeNode):
    """Check that the loss of a tree search node is accurate.

    Parameters
    ----------
    tasks : list of TaskRRM
    node : TreeNode

    """

    l_ex = node.l_ex
    l_eval = 0
    for n in range(len(tasks)):
        l_eval += tasks[n].loss_fcn(node.t_ex[n])
    if abs(l_eval - l_ex) > 1e-12:
        raise ValueError('Node loss is inaccurate.')


def branch_bound(tasks: list, n_ch, exhaustive=False, verbose=False, rng=rng_default):
    """Branch and Bound algorithm.
    
    Parameters
    ----------
    tasks : list of TaskRRM
    n_ch : int
        Number of channels.
    exhaustive : bool
        Enables exhaustive tree search.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks         # TODO: proper style to redefine class attribute here?
    TreeNode._n_ch = n_ch
    TreeNode._rng = rng

    stack = [TreeNodeBound([[] for _ in range(n_ch)])]      # Initialize Stack

    l_upper_min = stack[0].l_up

    # Iterate
    while not ((len(stack) == 1) and (len(stack[0].seq_rem) == 0)):
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_upper_min:.3f}')

        node = stack.pop()     # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True, exhaustive=exhaustive):
            # Bound
            if node_new.l_lo < l_upper_min:  # New node is not dominated
                if node_new.l_up < l_upper_min:
                    l_upper_min = node_new.l_up
                    stack = [s for s in stack if s.l_lo < l_upper_min]  # Cut Dominated Nodes

                if len(node_new.seq_rem) > 0:  # Add New Node to Stack
                    stack.append(node_new)     # LIFO
                else:
                    stack.insert(0, node_new)

    if len(stack) != 1:
        raise ValueError('Multiple nodes...')

    if not all([s.l_lo == s.l_up for s in stack]):
        raise ValueError('Node bounds do not converge.')

    _check_loss(tasks, stack[0])

    t_ex, ch_ex = stack[0].t_ex, stack[0].ch_ex      # optimal

    return t_ex, ch_ex


def mc_tree_search(tasks: list, n_ch, n_mc, verbose=False, rng=rng_default):
    """Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of TaskRRM
    n_ch : int
        Number of channels.
    n_mc : int
        Number of Monte Carlo roll-outs per task.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._n_ch = n_ch
    TreeNode._rng = rng

    node = TreeNode([[] for _ in range(n_ch)])

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

        ch_new = []
        lens = []
        for ch in range(n_ch):
            len_seq = len(node.seq[ch])
            lens.append(len_seq)
            if len(node_mc_best.seq[ch]) > len_seq:
                ch_new.append(ch)
        ch_app = rng.choice(ch_new)     # TODO: ad-hoc, randomly select appended channel for iteration

        seq_new = copy.deepcopy(node.seq)
        seq_new[ch_app].append(node_mc_best.seq[ch_app][lens[ch_app]])
        node.seq = seq_new      # call seq.setter

    _check_loss(tasks, node)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex

"""Multi-channel Tree Search objects and algorithms."""

import copy
import numpy as np

from sequence2schedule import FlexDARMultiChannelSequenceScheduler

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
    ch_avail : ndarray
        Channel availability times.
    l_ex : float
        Total loss of scheduled tasks.
    seq_rem: set
        Unscheduled task indices.

    """
    # TODO: docstring describes properties as attributes. OK? Subclasses, too.

    _tasks = []  # TODO: needs to be overwritten by invoking scripts...
    _ch_avail_init = []
    _exhaustive = False
    _rng = None

    def __init__(self, seq: list):
        if self._n_tasks == 0 or self._n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and '_n_ch class attributes.")

        self._seq = [[] for _ in range(self._n_ch)]

        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)  # task execution times (NaN for unscheduled)
        self._ch_ex = np.full(self._n_tasks, np.nan, dtype=np.int)

        self._ch_avail = np.zeros(self._n_ch)  # timeline availability

        self._l_ex = 0.  # partial sequence loss

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, partial loss:{self.l_ex:.3f})"

    @property
    def _n_tasks(self):
        return len(self._tasks)

    @property
    def _n_ch(self):
        return len(self._ch_avail_init)

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
    def t_ex(self):
        return self._t_ex

    @property
    def ch_ex(self):
        return self._ch_ex

    @property
    def ch_avail(self):
        return self._ch_avail

    @property
    def l_ex(self):
        return self._l_ex

    @property
    def seq_rem(self):
        return self._seq_rem

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
            if self._seq[ch] != seq[ch][:len(self._seq[ch])]:  # new sequence is not an extension of current sequence
                self.__init__(seq)  # initialize from scratch
                break

        for ch in range(self._n_ch):
            seq_append = seq[ch][len(self._seq[ch]):]  # new task indices to schedule
            self._seq_rem = self._seq_rem - set(seq_append)

            self._seq[ch] = seq[ch]

            self._ch_ex[seq_append] = ch
            for n in seq_append:  # recursively update Node attributes
                self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
                self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration
                self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True):
        """Generate descendant nodes.

        Parameters
        ----------
        do_permute : bool
            Enables random permutation of returned node list.

        Yields
        -------
        TreeNode
            Descendant node with one additional task scheduled.

        """

        seq_iter = list(self._seq_rem)
        if do_permute:
            self._rng.shuffle(seq_iter)

        if self._exhaustive:
            ch_iter = list(range(self._n_ch))  # try each task on each channel
        else:
            ch_iter = [int(np.argmin(self.ch_avail))]  # try each task on the earliest available channel only

        for n in seq_iter:

            if self._exhaustive and do_permute:
                self._rng.shuffle(ch_iter)

            for ch in ch_iter:
                seq_new = copy.deepcopy(self.seq)
                seq_new[ch].append(n)

                node_new = copy.deepcopy(self)  # new TreeNode object
                node_new.seq = seq_new  # call seq.setter method

                yield node_new

    def roll_out(self, do_copy=False):
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

        if self._exhaustive:
            _temp = self._rng.multinomial(self._n_tasks, np.ones(self._n_ch) / self._n_ch)
            splits = np.split(seq_rem_perm, np.cumsum(_temp)[:-1])
            for ch in range(self._n_ch):
                seq_new[ch].extend(splits[ch].tolist())

        else:
            ch_avail = copy.deepcopy(self.ch_avail)
            for n in seq_rem_perm:
                ch = int(np.argmin(ch_avail))
                seq_new[ch].append(n)

                ch_avail[ch] = max(self._tasks[n].t_release, ch_avail[ch]) + self._tasks[n].duration

        if do_copy:
            node_new = copy.deepcopy(self)  # new TreeNode object
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
        ch_avail : ndarray
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
    def l_lo(self):
        return self._l_lo

    @property
    def l_up(self):
        return self._l_up

    def update_node(self, seq: list):
        """Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.

        """

        super().update_node(seq)

        # Add bound attributes
        t_ex_max = (max([self._tasks[n].t_release for n in self._seq_rem] + list(self._ch_avail))
                    + sum([self._tasks[n].duration for n in self._seq_rem]))  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_release, min(self._ch_avail)))
            self._l_up += self._tasks[n].loss_fcn(t_ex_max)

        if len(self._seq_rem) > 0 and self._l_lo == self._l_up:  # roll-out if bounds converge
            self.roll_out()


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


def branch_bound(tasks: list, ch_avail: list, exhaustive=False, verbose=False, rng=rng_default):
    """Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.
    exhaustive : bool
        Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.
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

    TreeNode._tasks = tasks  # TODO: proper style to redefine class attribute here?
    TreeNode._ch_avail_init = ch_avail
    TreeNode._exhaustive = exhaustive
    TreeNode._rng = rng

    n_ch = len(ch_avail)

    stack = [TreeNodeBound([[] for _ in range(n_ch)])]  # Initialize Stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution
    l_best = node_best.l_ex

    # Iterate
    while len(stack) > 0:
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < l_best:  # New node is not dominated
                if node_new.l_up < l_best:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    l_best = node_best.l_ex
                    stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

    _check_loss(tasks, node_best)

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    return t_ex, ch_ex


def mc_tree_search(tasks: list, ch_avail: list, n_mc, exhaustive=False, verbose=False, rng=rng_default):
    """Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.
    n_mc : int
        Number of Monte Carlo roll-outs per task.
    exhaustive : bool
        Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.
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
    TreeNode._ch_avail_init = ch_avail
    TreeNode._exhaustive = exhaustive
    TreeNode._rng = rng

    n_ch = len(ch_avail)

    node = TreeNode([[] for _ in range(n_ch)])
    node_best = node.roll_out(do_copy=True)

    n_tasks = len(tasks)
    for n in range(n_tasks):
        if verbose:
            print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')

        # Perform Roll-outs
        for _ in range(n_mc):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_best.l_ex:  # Update best node
                node_best = node_mc

        seq_new = copy.deepcopy(node.seq)

        # Assign next task from earliest available channel
        ch = int(np.argmin(node.ch_avail))
        seq_new[ch].append(node_best.seq[ch][len(node.seq[ch])])

        node.seq = seq_new  # call seq.setter

    _check_loss(tasks, node)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def EstAlg(tasks: list, ch_avail: list, exhaustive=False, verbose=False, rng=rng_default):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.
    exhaustive : bool
        Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.
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

    N = len(tasks)
    n_tasks = len(tasks)
    n_ch = len(ch_avail)
    t_release = np.zeros(N)
    for n in range(n_tasks):
        t_release[n] = tasks[n].t_release
        # t_release.append(tasks[n].t_release)

    #a = 2
    T = np.argsort(t_release)  # Task Order
    RP = 100
    t_ex, ch_ex = FlexDARMultiChannelSequenceScheduler(T, tasks, ch_avail, RP)
    #   t_ex = np.sort(t_release)
    #   ch_ex = []

    # Assign next task from earliest available channel
    # ch = int(np.argmin(node.ch_avail))
    # seq_new[ch].append(node_best.seq[ch][len(node.seq[ch])])

    return t_ex, ch_ex

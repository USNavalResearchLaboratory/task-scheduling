"""Multi-channel Tree Search objects and algorithms."""

import copy
import numpy as np
from util.utils import check_rng

from sequence2schedule import FlexDARMultiChannelSequenceScheduler


class TreeNode:
    """Node object for tree search algorithms.

    Parameters
    ----------
    seq : list of list
        List of task index sequences by channel

    Attributes
    ----------
    seq : list of int
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

    _tasks = []  # TODO: needs to be overwritten by invoking scripts... OK?
    _ch_avail_init = []
    _rng = None

    def __init__(self, seq: list):
        if self._n_tasks == 0 or self._n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and '_n_ch class attributes.")

        self._seq = []

        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)  # task execution times (NaN for unscheduled)
        self._ch_ex = np.full(self._n_tasks, np.nan, dtype=np.int)  # task execution channels

        self._ch_avail = copy.deepcopy(self._ch_avail_init)  # timeline availability

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
        list of int
            Task index sequence

        """
        return self._seq

    @seq.setter
    def seq(self, seq):
        if len(seq) != len(set(seq)):
            raise ValueError("Input 'seq' must have unique values.")

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
    def ch_early(self):
        return int(np.argmin(self.ch_avail))

    @property
    def l_ex(self):
        return self._l_ex

    @property
    def seq_rem(self):
        return self._seq_rem

    def update_node(self, seq: list):
        """Sets node sequence using sequence-to-schedule approach.

        Parameters
        ----------
        seq : list of int
            Sequence of indices referencing cls._tasks.

        """

        if seq[:len(self._seq)] != self._seq:  # new sequence is not an extension of current sequence
            self.__init__(seq)  # initialize from scratch

        seq_append = seq[len(self._seq):]
        self._seq = seq
        self._seq_rem -= set(seq_append)
        for n in seq_append:
            ch = self.ch_early

            self._ch_ex[n] = ch
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

        for n in seq_iter:
            seq_new = copy.deepcopy(self.seq)
            seq_new.append(n)

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

        seq_new = copy.deepcopy(self.seq) + self._rng.permutation(list(self._seq_rem)).tolist()

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
        seq : list of int
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


def branch_bound(tasks: list, ch_avail: list, verbose=False, rng=None):
    """Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of BaseTask
    ch_avail : list of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    stack = [TreeNodeBound([])]  # Initialize Stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution
    l_best = node_best.l_ex

    # Iterate
    while len(stack) > 0:
        if verbose:
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):  # TODO: check cutting! inequality?
            # Bound
            if node_new.l_lo < l_best:  # New node is not dominated
                if node_new.l_up < l_best:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    l_best = node_best.l_ex
                    stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    return t_ex, ch_ex


def mc_tree_search(tasks: list, ch_avail: list, n_mc, verbose=False, rng=None):
    """Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of BaseTask
    ch_avail : list of float
        Channel availability times.
    n_mc : int
        Number of Monte Carlo roll-outs per task.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    node = TreeNode([])
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

        # Assign next task from earliest available channel
        seq_new = copy.deepcopy(node.seq)
        seq_new.append(node_best.seq[len(node.seq)])
        node.seq = seq_new  # call seq.setter

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def random_sequencer(tasks: list, ch_avail: list, rng=None):
    """Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : list of BaseTask
    ch_avail : list of float
        Channel availability times.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    node = TreeNode([]).roll_out(do_copy=True)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def est_alg(tasks: list, ch_avail: list):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of BaseTask
    ch_avail : list of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = np.argsort([task.t_release for task in tasks]).tolist()
    node = TreeNode(seq)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def EstAlg(tasks: list, ch_avail: list):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of BaseTask
    ch_avail : list of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    t_release = [task.t_release for task in tasks]

    # a = 2
    T = np.argsort(t_release)  # Task Order
    RP = 100
    ChannelAvailableTime = copy.deepcopy(ch_avail)
    t_ex, ch_ex = FlexDARMultiChannelSequenceScheduler(T, tasks, ChannelAvailableTime, RP)
    #   t_ex = np.sort(t_release)
    #   ch_ex = []

    # Assign next task from earliest available channel
    # ch = int(np.argmin(node.ch_avail))
    # seq_new[ch].append(node_best.seq[ch][len(node.seq[ch])])

    return t_ex, ch_ex


def est_task_swap_alg(tasks: list, ch_avail: list):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = np.argsort([task.t_release for task in tasks]).tolist()
    node = TreeNode(seq)
    N = len(seq)

    for jj in range(N - 1):  #
        Tswap = copy.deepcopy(seq)
        T1 = seq[jj]
        T2 = seq[jj + 1]
        Tswap[jj] = T2
        Tswap[jj + 1] = T1
        nodeSwap = TreeNode(Tswap)
        if nodeSwap.l_ex < node.l_ex:
            seq = copy.deepcopy(Tswap)
            node = TreeNode(seq)
            # breakpoint()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def ed_alg(tasks: list, ch_avail: list):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = np.argsort([task.t_drop for task in tasks]).tolist()
    node = TreeNode(seq)

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def ed_swap_task_alg(tasks: list, ch_avail: list):
    """Earliest Start Times Algorithm

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail : list of float
        Channel availability times.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = np.argsort([task.t_drop for task in tasks]).tolist()
    node = TreeNode(seq)
    N = len(seq)

    for jj in range(N - 1):  #
        Tswap = copy.deepcopy(seq)
        T1 = seq[jj]
        T2 = seq[jj + 1]
        Tswap[jj] = T2
        Tswap[jj + 1] = T1
        nodeSwap = TreeNode(Tswap)
        if nodeSwap.l_ex < node.l_ex:
            seq = copy.deepcopy(Tswap)
            node = TreeNode(seq)
            # breakpoint()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex
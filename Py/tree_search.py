"""Multi-channel Tree Search objects and algorithms."""

from copy import deepcopy
import math
import numpy as np
from util.generic import check_rng

from tasks import ReluDropGenerator

from sequence2schedule import FlexDARMultiChannelSequenceScheduler


class TreeNode:
    """
    Node object for tree search algorithms.

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

    def __init__(self, seq=None):
        if seq is None:
            seq = []

        if self._n_tasks == 0 or self._n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and '_n_ch class attributes.")

        self._seq = []

        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)  # task execution times (NaN for unscheduled)
        self._ch_ex = np.full(self._n_tasks, -1)    # TODO: masked array?

        self._ch_avail = self._ch_avail_init.copy()     # timeline availability

        self._l_ex = 0.  # partial sequence loss

        self.seq_extend(seq)

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
        """
        Gets the node sequence. Setter calls '__init__' or 'seq_ext'.

        Returns
        -------
        list of int
            Task index sequence

        """
        return self._seq

    @seq.setter
    def seq(self, seq):
        if seq[:len(self._seq)] != self._seq:  # new sequence is not an extension of current sequence
            self.__init__(seq)  # initialize from scratch
        else:
            self.seq_extend(seq[len(self._seq):])

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

    def seq_extend(self, seq_ext: list):
        """
        Extends node sequence and updates attributes using sequence-to-schedule approach.

        Parameters
        ----------
        seq_ext : list of int
            Sequence of indices referencing cls._tasks.

        """

        seq_ext_set = set(seq_ext)
        if len(seq_ext) != len(seq_ext_set):
            raise ValueError("Input 'seq_ext' must have unique values.")
        if not seq_ext_set.issubset(self.seq_rem):
            raise ValueError("Values in 'seq_ext' must not be in the current node sequence.")

        self._seq_extend(seq_ext)   # TODO: just do unique checks based on input flag? append method?

    def _seq_extend(self, seq_ext: list):
        self._seq += seq_ext
        self._seq_rem -= set(seq_ext)

        for n in seq_ext:
            ch = self.ch_early

            self._ch_ex[n] = ch
            self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
            self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration
            self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True):
        """
        Generate descendant nodes.

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
            node_new = deepcopy(self)  # new TreeNode object
            node_new._seq_extend([n])
            yield node_new

    def roll_out(self, do_copy=False):
        """
        Generates/updates node with a randomly completed sequence.

        Parameters
        ----------
        do_copy : bool
            Enables return of a new TreeNode object. Otherwise, updates in-place.

        Returns
        -------
        TreeNode
            Only if do_copy is True.

        """

        seq_ext = self._rng.permutation(list(self._seq_rem)).tolist()

        if do_copy:
            node_new = deepcopy(self)  # new TreeNode object
            node_new._seq_extend(seq_ext)
            return node_new
        else:
            self._seq_extend(seq_ext)

    def check_swaps(self):
        """Try adjacent task swapping, overwrite node if loss drops."""

        if len(self.seq_rem) != 0:
            raise ValueError("Node sequence must be complete.")

        for i in range(len(self.seq) - 1):
            seq_swap = self.seq.copy()
            seq_swap[i:i + 2] = seq_swap[i:i + 2][::-1]
            node_swap = TreeNode(seq_swap)
            if node_swap.l_ex < self.l_ex:
                self = node_swap            # TODO: improper?


class TreeNodeBound(TreeNode):
    """
    Node object with additional loss bounding attributes.

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

    def __init__(self, seq=None):
        if seq is None:
            seq = []

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

    def _seq_extend(self, seq: list):
        """
        Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.

        """

        super()._seq_extend(seq)

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


class SearchTree:

    def __init__(self, rng=None):
        self.root = TreeNode()
        self.rng = check_rng(rng)

        self.n_tasks = self.root._n_tasks

        class SearchNode:

            rng = check_rng(None)  # TODO: move, delete?
            n_tasks = self.root._n_tasks

            def __init__(self, seq=None):
                if seq is None:
                    seq = []

                self.seq = seq
                self.n_visit = 0
                self.l_avg = 0.
                self.children = {}

            def __repr__(self):
                return f"SearchNode({self.seq}, {self.n_visit}, {self.l_avg:.3f})"

            def __getitem__(self, n):
                return self.children[n]

            # def __setitem__(self, key, value):
            #     self.children[key] = value

            @property
            def seq_rem(self):
                return set(range(self.n_tasks)) - set(self.seq)

            @property
            def is_leaf(self):
                return len(self.seq_rem) == 0

            def add_child(self, n):
                self.children[n] = SearchNode(self.seq + [n])

            @staticmethod
            def stat2weight(n_visit, l_avg):
                return l_avg - 1 / (n_visit + 1)

            def select_child(self):
                w = {k: self.stat2weight(node.n_visit, node.l_avg) for (k, node) in self.children.items()}
                seq_rem = SearchNode.rng.permutation(list(self.seq_rem))
                w.update(dict(zip(seq_rem, [-1] * len(seq_rem))))

                n = min(w, key=w.__getitem__)
                if n not in self.children:
                    self.add_child(n)
                return self[n]

            def simulate(self):
                node = self
                # seq = []
                while not node.is_leaf:
                    node = node.select_child()
                    # seq.append(n)
                    # node = node[n]

                return node.seq

        # self.cls_node = SearchNode

        # self.tree = [{} for _ in range(self.n_tasks)]
        self.tree = SearchNode()

    def __getitem__(self, seq):
        out = self.tree
        for n in seq:
            out = out[n]
        return out

    # def __getitem__(self, item):
    #     out = self.tree[len(item) - 1]
    #     for n in item:
    #         out = out[n]
    #     return out

    # def _seq_rem(self, seq):
    #     return set(range(self.n_tasks)) - set(seq)

    # def select(self, d):
    #     # w = dict(zip(s.keys(), map(self.stat2weight, s.values())))
    #     w = {k: self.stat2weight(*v) for (k, v) in d.items()}
    #     # seq_rem = self._seq_rem(d.keys())    # TODO: permute dict for random tie-breaks?
    #     seq_rem = self.rng.permutation(list(self._seq_rem(d.keys())))
    #     w.update(dict(zip(seq_rem, [-1] * len(seq_rem))))    # FIXME
    #
    #     return min(w, key=w.__getitem__)

    # def rollout(self):
    #     seq = []
    #     for i in range(self.n_tasks):
    #         d = self.tree[i]
    #         for n in seq:
    #             d = d[n]
    #         n = self.select(d)
    #         if n not in d.keys():     # initialize tree element
    #             d[n] = (0, 0)
    #
    #         seq.append(n)





    # def _br(self, node):
    #     return dict(zip(node.seq_rem, node.branch(do_permute=False)))
    #
    # def branch(self, seq):
    #     if len(seq) == 0:
    #         self.tree = self._br(self.tree)
    #     else:
    #         tree = self.tree
    #         for n in seq[:-1]:
    #             tree = tree[n]
    #         tree[seq[-1]] = self._br(tree[seq[-1]])

    # TODO: WIP, delete?


def branch_bound(tasks: list, ch_avail: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
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

    stack = [TreeNodeBound()]  # Initialize Stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < node_best.l_ex:  # New node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            # progress = 1 - sum([math.factorial(len(node.seq_rem)) for node in stack]) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {node_best.l_ex:.3f}', end='\r')

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    return t_ex, ch_ex


def branch_bound_with_stats(tasks: list, ch_avail: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
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
    NodeStats = [TreeNodeBound([])]
    # NodeStats = []

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution
    l_best = node_best.l_ex
    NodeStats.append(node_best)

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # Extract Node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if len(node_new.seq) == len(tasks):
                NodeStats.append(node_new) # Append any complete solutions, use for training NN. Can decipher what's good/bad based on final costs

            if node_new.l_lo < l_best:  # New node is not dominated
                if node_new.l_up < l_best:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    # NodeStats.append(node_best)
                    l_best = node_best.l_ex
                    stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            # progress = 1 - sum([math.factorial(len(node.seq_rem)) for node in stack]) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {l_best:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal
    NodeStats.pop(0) # Remove First Initialization stage
    return t_ex, ch_ex, NodeStats


def mcts(tasks: list, ch_avail: list, n_mc, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    n_mc : int or list of int
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

    node = TreeNode()
    node_best = node.roll_out(do_copy=True)

    n_tasks = len(tasks)
    if type(n_mc) == int:
        n_mc = n_tasks * [n_mc]

    for n in range(n_tasks):
        if verbose:
            print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')

        # Perform Roll-outs
        for _ in range(n_mc[n]):       # TODO: variable number of roll-outs by stage for efficiency
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_best.l_ex:  # Update best node
                node_best = node_mc

        # Assign next task from earliest available channel
        node._seq_extend([node_best.seq[n]])

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def mcts_v2(tasks: list, ch_avail: list, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    n_mc : int or list of int
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

    # TODO: edit docstring

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    rng = check_rng(rng)
    TreeNode._rng = rng

    n_tasks = len(tasks)

    # def _br(self, node):
    #     return dict(zip(node.seq_rem, node.branch(do_permute=False)))
    #
    # def branch(self, seq):
    #     if len(seq) == 0:
    #         self.tree = self._br(self.tree)
    #     else:
    #         tree = self.tree
    #         for n in seq[:-1]:
    #             tree = tree[n]
    #         tree[seq[-1]] = self._br(tree[seq[-1]])

    def _update_stat(seq, loss):
        pass

    def rand_select(seq):
        s = stats[len(seq)]
        for n in seq:
            s = s[n]


    # Initialize tree
    tree = TreeNode()
    stats = [{} for _ in range(n_tasks)]

    seq = []
    node = tree
    n = rng.choice(list(node.seq_rem))
    seq.append(n)
    for n in range(1, n_tasks):
        node = tree[seq[0]]
        for i in range(1, n):
            node = node[seq[i]]

        node = dict(zip(node.seq_rem, node.branch(do_permute=False)))
        # node = tree[n_rand]


    do_search = True
    while do_search:
        node = tree[0]
        # Selection
        for n in range(n_tasks):
            #
            if node.is_leaf:
                break

        # Expansion
        _temp = dict(zip(node.seq_rem, node.branch(do_permute=False)))
        tree[n].append(_temp)







def random_sequencer(tasks: list, ch_avail: list, rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : list of GenericTask
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


def earliest_release(tasks: list, ch_avail: list, do_swap=False):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    do_swap : bool
        Enables task swapping

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_release for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def earliest_drop(tasks: list, ch_avail: list, do_swap=False):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : list of ReluDropTask
    ch_avail : list of float
        Channel availability times.
    do_swap : bool
        Enables task swapping.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_drop for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    t_ex, ch_ex = node.t_ex, node.ch_ex

    return t_ex, ch_ex


def est_alg_kw(tasks: list, ch_avail: list):
    """
    Earliest Start Times Algorithm using FlexDAR scheduler function.

    Parameters
    ----------
    tasks : list of GenericTask
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

    T = np.argsort(t_release)  # Task Order
    RP = 100
    ChannelAvailableTime = deepcopy(ch_avail)
    t_ex, ch_ex = FlexDARMultiChannelSequenceScheduler(T, tasks, ChannelAvailableTime, RP)

    return t_ex, ch_ex


if __name__ == '__main__':
    n_tasks = 3
    n_channels = 2

    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=None)  # task set generator

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 2, n_ch)

    tasks = task_gen.rand_tasks(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    TreeNode._tasks = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(None)

    # node = TreeNode([3, 1])
    # node.seq = [3, 1, 4]

    # t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True, rng=None)

    tree = SearchTree()
    node = tree[[]]
    node.add_child(1)
    node.select_child()
    seq = node.simulate()
    pass

# if __name__ == '__main__':
#     main()

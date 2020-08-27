"""Multi-channel Tree Search objects and algorithms."""

from copy import deepcopy

import numpy as np

from util.generic import check_rng
# from generators.scheduling_problems import Random as RandomProblem

from sequence2schedule import FlexDARMultiChannelSequenceScheduler

np.set_printoptions(precision=2)


class TreeNode:
    """
    Node object for mapping sequences into optimal execution times and channels.

    Parameters
    ----------
    seq : Sequence of int
        Partial task index sequence.

    Attributes
    ----------
    seq : Sequence of int
        Partial task index sequence.
    t_ex : ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : ndarray
        Task execution channels. -1 for unscheduled.
    ch_avail : ndarray
        Channel availability times.
    l_ex : float
        Total loss of scheduled tasks.
    seq_rem: set
        Unscheduled task indices.

    """

    _tasks_init = ()    # TODO: needs to be overwritten by invoking scripts... OK?
    _ch_avail_init = ()
    _rng = None

    def __init__(self, seq=None):
        self._tasks = deepcopy(self._tasks_init)
        self._ch_avail = np.array(self._ch_avail_init).copy()  # timeline availability

        if self.n_tasks == 0 or self.n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and 'n_ch class attributes.")
        elif min(self._ch_avail) < 0.:
            raise ValueError("Initial channel availabilities must be non-negative.")

        self._seq = []
        self._seq_rem = set(range(self.n_tasks))

        self._t_ex = np.full(self.n_tasks, np.nan)
        self._ch_ex = np.full(self.n_tasks, -1)

        self._l_ex = 0.  # incurred loss

        if seq is not None and len(seq) > 0:
            self.seq_extend(seq)

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    def summary(self):
        """Print a string describing important node attributes."""
        print(f'TreeNode\n--------\nsequence: {self.seq}\nexecution time: {self.t_ex}'
              f'\nexecution channel: {self.ch_ex}\nloss incurred: {self.l_ex:.2f}')

    tasks = property(lambda self: self._tasks)
    ch_avail = property(lambda self: self._ch_avail)
    n_tasks = property(lambda self: len(self._tasks))
    n_ch = property(lambda self: len(self._ch_avail))

    seq_rem = property(lambda self: self._seq_rem)
    ch_min = property(lambda self: int(np.argmin(self.ch_avail)))

    t_ex = property(lambda self: self._t_ex)
    ch_ex = property(lambda self: self._ch_ex)
    l_ex = property(lambda self: self._l_ex)

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

    def seq_extend(self, seq_ext, check_valid=True):
        """
        Extends node sequence and updates attributes using sequence-to-schedule approach.

        Parameters
        ----------
        seq_ext : int or Sequence of int
            Sequence of indices referencing cls._tasks.
        check_valid : bool
            Perform check of index sequence validity.

        """

        if isinstance(seq_ext, (int, np.integer)):
            seq_ext = [seq_ext]
        if len(seq_ext) == 0:
            return

        if check_valid:
            seq_ext_set = set(seq_ext)
            if len(seq_ext) != len(seq_ext_set):
                raise ValueError("Input 'seq_ext' must have unique values.")
            if not seq_ext_set.issubset(self._seq_rem):
                raise ValueError("Values in 'seq_ext' must not be in the current node sequence.")

        for n in seq_ext:
            self._seq.append(n)
            self._seq_rem.remove(n)
            self._update_ex(n, self.ch_min)     # assign task to channel with earliest availability

    def _update_ex(self, n, ch):
        self._ch_ex[n] = ch

        self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
        self._l_ex += self._tasks[n](self._t_ex[n])     # add task execution loss

        self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration    # new channel availability

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
            node_new.seq_extend(n, check_valid=False)
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
            node_new.seq_extend(seq_ext, check_valid=False)
            return node_new
        else:
            self.seq_extend(seq_ext, check_valid=False)

    def check_swaps(self):
        """Try adjacent task swapping, overwrite node if loss drops."""

        if len(self._seq_rem) != 0:
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
    seq : list of int
        Partial task index sequence.

    Attributes
    ----------
    seq : list of int
        Partial task index sequence.
    t_ex : ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : ndarray
        Task execution channels. -1 for unscheduled.
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
        self._l_lo = 0.
        self._l_up = float('inf')
        super().__init__(seq)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    l_lo = property(lambda self: self._l_lo)
    l_up = property(lambda self: self._l_up)

    def seq_extend(self, seq: list, check_valid=True):
        """
        Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.
        check_valid : bool
            Perform check of index sequence validity.

        """

        super().seq_extend(seq, check_valid)

        # Add bound attributes
        t_ex_max = (max([self._tasks[n].t_release for n in self._seq_rem] + [min(self._ch_avail)])
                    + sum(self._tasks[n].duration for n in self._seq_rem))  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n](max(self._tasks[n].t_release, min(self._ch_avail)))
            self._l_up += self._tasks[n](t_ex_max)

        # Roll-out if bounds converge
        if len(self._seq_rem) > 0 and self._l_lo == self._l_up:
            self.roll_out()


class TreeNodeShift(TreeNode):
    def __init__(self, seq=None):
        self.t_origin = 0.
        super().__init__(seq)

        if len(self._seq) == 0:     # otherwise, shift_origin is invoked during initial call to 'seq_extend'
            self.shift_origin()

    def __repr__(self):
        return f"TreeNodeShift(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    def shift_origin(self):
        """Shifts the time origin to the earliest channel availability and invokes shift method of each task,
        adding each incurred loss to the total."""

        ch_avail_min = min(self._ch_avail)
        if ch_avail_min == 0.:
            return

        self.t_origin += ch_avail_min
        self._ch_avail -= ch_avail_min
        for n, task in enumerate(self._tasks):
            loss_inc = task.shift_origin(ch_avail_min)
            if n in self._seq_rem:
                self._l_ex += loss_inc      # add loss incurred due to origin shift for any unscheduled tasks

    def _update_ex(self, n, ch):
        self._ch_ex[n] = ch

        t_ex_rel = max(self._tasks[n].t_release, self._ch_avail[ch])  # relative to time origin
        self._t_ex[n] = self.t_origin + t_ex_rel    # absolute time
        self._l_ex += self._tasks[n](t_ex_rel)      # add task execution loss

        self._ch_avail[ch] = t_ex_rel + self._tasks[n].duration  # relative to time origin
        self.shift_origin()


class SearchNode:
    """
    Node object for Monte Carlo tree search.

    Parameters
    ----------
    seq : list of int
        Partial task index sequence.

    Attributes
    ----------
    seq : list of int
        Partial task index sequence.
    n_visits : int
        Number of times a roll-out has passed through the node.
    l_avg : float
        Average execution loss of roll-outs passing through the node.
    children : dict of {int: SearchNode}
        Channel availability times.
    weight : float
        Weighting for minimization objective during child node selection.

    """

    n_tasks = None
    l_up = None

    def __init__(self, seq=None):
        if seq is None:
            seq = []
        self._seq = seq

        self._n_visits = 0
        self._l_avg = 0.
        self._l_min = float('inf')  # FIXME: min? ordered statistic?
        self._children = {}

        self._seq_unk = set(range(self.n_tasks)) - set(seq)     # set of unexplored task indices

    def __repr__(self):
        return f"SearchNode(seq={self._seq}, children={list(self._children.keys())}, " \
               f"visits={self._n_visits}, avg_loss={self._l_avg:.3f})"

    def __getitem__(self, item):
        """
        Access a descendant node.

        Parameters
        ----------
        item : int or list of int
            Index of sequence of indices for recursive child node selection.

        Returns
        -------
        SearchNode

        """

        if type(item) == int:
            return self._children[item]
        else:
            node = self
            for n in item:
                node = node._children[n]
            return node

    seq = property(lambda self: self._seq)
    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)
    children = property(lambda self: self._children)

    @property
    def weight(self):
        return self._l_avg - 10 / (self._n_visits + 1)       # FIXME: placeholder. Need real metric and user control.
        # return np.random.random()

    def select_child(self):
        """
        Select a child node according to exploration/exploitation objective minimization.

        Returns
        -------
        SearchNode

        """

        # TODO: learn selection function with value network? Fast EST based selection?
        # TODO: add epsilon-greedy selector?

        w = {n: node.weight for (n, node) in self._children.items()}
        w.update({n: -10 for n in self._seq_unk})   # FIXME: value? random permute?

        n = min(w, key=w.__getitem__)
        if n not in self._children:
            self._children[n] = SearchNode(self._seq + [n])
            self._seq_unk.remove(n)

        return self[n]

    def simulate(self):
        """
        Produce an index sequence from iterative child node selection.

        Returns
        -------
        list of int

        """

        node = self
        while len(node._seq) < SearchNode.n_tasks:
            node = node.select_child()
        return node._seq

    def update_stats(self, loss):
        """
        Update visit count and average loss of roll-outs.

        Parameters
        ----------
        loss : float
            Loss of a complete solution descending from the node.

        """

        loss_total = self._l_avg * self._n_visits + loss
        self._n_visits += 1
        self._l_avg = loss_total / self._n_visits

    def backup(self, seq: list, loss):
        """
        Updates search attributes for all descendant nodes corresponding to an index sequence.

        Parameters
        ----------
        seq : list of int
            Complete task index sequence.
        loss : float
            Loss of a complete solution descending from the node.

        """

        if len(seq) != SearchNode.n_tasks:
            raise ValueError('Sequence must be complete.')

        seq_rem = seq[len(self._seq):]

        node = self
        node.update_stats(loss)
        for n in seq_rem:
            node = node[n]
            node.update_stats(loss)


def branch_bound(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    stack = [TreeNodeBound()]  # initialize stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # extract node

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < node_best.l_ex:  # new node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # cut dominated nodes

                stack.append(node_new)  # add new node to stack, LIFO

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

    return node_best.t_ex, node_best.ch_ex  # optimal


def branch_bound_with_stats(tasks, ch_avail, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    stack = [TreeNodeBound()]  # Initialize Stack
    NodeStats = [TreeNodeBound()]
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
                # Append any complete solutions, use for training NN. Can decipher what's good/bad based on final costs
                NodeStats.append(node_new)

            if node_new.l_lo < l_best:  # New node is not dominated
                if node_new.l_up < l_best:
                    node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                    # NodeStats.append(node_best)
                    l_best = node_best.l_ex
                    stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {l_best:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

    NodeStats.pop(0)    # Remove First Initialization stage
    return node_best.t_ex, node_best.ch_ex, NodeStats


def mcts_orig(tasks, ch_avail, n_mc, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
        Channel availability times.
    n_mc : int or Sequence of int
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

    TreeNode._tasks_init = tasks
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
        for _ in range(n_mc[n]):
            node_mc = node.roll_out(do_copy=True)

            if node_mc.l_ex < node_best.l_ex:  # Update best node
                node_best = node_mc

        # Assign next task from earliest available channel
        node.seq_extend(node_best.seq[n], check_valid=False)

    return node_best.t_ex, node_best.ch_ex


def mcts(tasks, ch_avail, n_mc, verbose=False):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
        Channel availability times.
    n_mc : int
        Number of roll-outs performed.
    verbose : bool
        Enables printing of algorithm state information.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    # TODO: add exploration/exploitation input control.

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    # TreeNode._rng = check_rng(rng)

    SearchNode.n_tasks = len(tasks)
    SearchNode.l_up = TreeNodeBound().l_up
    tree = SearchNode()

    node_best = None

    loss_min = float('inf')
    do_search = True
    while do_search:
        if verbose:
            print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')

        seq = tree.simulate()   # Roll-out a complete sequence
        node = TreeNode(seq)    # Evaluate execution times and channels, total loss

        loss = node.l_ex
        tree.backup(seq, loss)  # Update search tree from leaf sequence to root

        if loss < loss_min:
            node_best, loss_min = node, loss

        do_search = tree.n_visits < n_mc

    return node_best.t_ex, node_best.ch_ex


def random_sequencer(tasks, ch_avail, rng=None):
    """
    Generates a random task sequence, determines execution times and channels.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    node = TreeNode().roll_out(do_copy=True)

    return node.t_ex, node.ch_ex


def earliest_release(tasks, ch_avail, do_swap=False):
    """
    Earliest Start Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_release for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    return node.t_ex, node.ch_ex


def earliest_drop(tasks, ch_avail, do_swap=False):
    """
    Earliest Drop Times Algorithm.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_drop for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    return node.t_ex, node.ch_ex


def est_alg_kw(tasks, ch_avail):
    """
    Earliest Start Times Algorithm using FlexDAR scheduler function.

    Parameters
    ----------
    tasks : Sequence of tasks.Generic
    ch_avail : Sequence of float
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


def ert_alg_kw(tasks, ch_avail, do_swap=False):

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail

    seq = list(np.argsort([task.t_release for task in tasks]))
    node = TreeNode(seq)

    if do_swap:
        node.check_swaps()

    t_ex, ch_ex, T = node.t_ex, node.ch_ex, node.seq

    return t_ex, ch_ex, T


def main():
    n_tasks = 5
    n_channels = 2

    problem_gen = RandomProblem.relu_drop_default(n_tasks, n_channels)
    (tasks, ch_avail), = problem_gen()

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(None)

    seq = [3, 1, 4]
    seq = np.random.permutation(n_tasks)
    node, node_s = TreeNode(seq), TreeNodeShift(seq)
    print(node.t_ex)
    print(node_s.t_ex)

    t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True, rng=None)

    SearchNode.n_tasks = n_tasks
    SearchNode.l_up = TreeNodeBound().l_up

    node = SearchNode()
    child = node.select_child()
    leaf = node.simulate()
    pass

    t_ex, ch_ex = mcts_orig(tasks, ch_avail, n_mc=[1000 for n in range(n_tasks, 0, -1)], verbose=False)
    print(t_ex)
    t_ex, ch_ex = mcts(tasks, ch_avail, verbose=False)
    print(t_ex)


if __name__ == '__main__':
    main()

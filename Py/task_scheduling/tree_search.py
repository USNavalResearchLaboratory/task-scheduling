from collections import deque
from copy import deepcopy
from math import factorial
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd

from task_scheduling.tasks import Base as BaseTask, Shift as ShiftTask
from task_scheduling.util.generic import RandomGeneratorMixin


# TODO: modify classes and algorithms to efficiently handle repeated tasks!?


class TreeNode(RandomGeneratorMixin):
    """
    Node object for mapping sequences into optimal execution times and channels.

    Parameters
    ----------
    tasks : Sequence of BaseTask
    ch_avail : Sequence of float
        Channel availabilities
    seq : Sequence of int
        Partial task index sequence.
    rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

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

    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        super().__init__(rng)

        self._tasks = deepcopy(tasks)  # TODO: slow? add option for copy?
        self._ch_avail = np.array(ch_avail, dtype=float)

        if min(self._ch_avail) < 0.:
            raise ValueError("Initial channel availabilities must be non-negative.")

        self._seq = []
        self._seq_rem = set(range(self.n_tasks))

        self._t_ex = np.full(self.n_tasks, np.nan)
        self._ch_ex = np.full(self.n_tasks, -1)

        self._l_ex = 0.  # incurred loss

        if len(seq) > 0:
            self.seq_extend(seq)

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return (self.tasks, self.ch_avail, self.seq) == (other.tasks, other.ch_avail, other.seq)
        else:
            return NotImplemented

    def summary(self, file=None):
        """Print a string describing important node attributes."""
        keys = ('seq', 't_ex', 'ch_ex', 'l_ex')
        df = pd.Series({key: getattr(self, key) for key in keys})
        print(df.to_markdown(tablefmt='github', floatfmt='.3f'), file=file)

        # str_out = f'TreeNode\n- sequence: {self.seq}\n- execution times: {self.t_ex}' \
        #           f'\n- execution channels: {self.ch_ex}\n- loss incurred: {self.l_ex:.3f}'
        # print(str_out)
        # return str_out

    tasks = property(lambda self: self._tasks)
    ch_avail = property(lambda self: self._ch_avail)
    n_tasks = property(lambda self: len(self._tasks))
    n_ch = property(lambda self: len(self._ch_avail))

    seq_rem = property(lambda self: self._seq_rem)
    ch_min = property(lambda self: np.argmin(self.ch_avail))

    t_ex = property(lambda self: self._t_ex)
    ch_ex = property(lambda self: self._ch_ex)
    l_ex = property(lambda self: self._l_ex)

    @property
    def seq(self):
        """Task index sequence."""
        return self._seq

    @seq.setter
    def seq(self, seq):
        seq_prev, seq_ext = seq[:len(self._seq)], seq[len(self._seq):]
        if seq_prev == self._seq:  # new sequence is an extension of current sequence
            if len(seq_ext) > 0:
                self.seq_extend(seq_ext)
        else:
            # self.__init__(self.tasks, self.ch_avail, seq, rng=self.rng)  # initialize from scratch
            # TODO: shift nodes cannot recover original task/ch_avail state!
            raise ValueError(f"Sequence must be an extension of {self._seq}")

    def seq_extend(self, seq_ext, check_valid=True):
        """
        Extends node sequence and updates attributes using sequence-to-schedule approach.

        Parameters
        ----------
        seq_ext : int or Sequence of int
            Indices referencing self.tasks.
        check_valid : bool
            Perform check of index sequence validity.

        """

        if isinstance(seq_ext, Integral):
            self.seq_append(seq_ext, check_valid)
        else:
            if check_valid:
                set_ext = set(seq_ext)
                if len(seq_ext) != len(set_ext):
                    raise ValueError("Input 'seq_ext' must have unique values.")
                elif not set_ext.issubset(self._seq_rem):
                    raise ValueError("Values in 'seq_ext' must not be in the current node sequence.")

            for n in seq_ext:
                self.seq_append(n, check_valid=False)

    def seq_append(self, n, check_valid=True):
        """
        Extends node sequence and updates attributes using sequence-to-schedule approach.

        Parameters
        ----------
        n : Integral
            Index referencing self.tasks.
        check_valid : bool
            Perform check of index validity.

        """

        if check_valid and n in self.seq:
            raise ValueError("Appended index must not be in the current node sequence.")

        self._seq.append(n)
        self._seq_rem.remove(n)
        self._update_ex(n, self.ch_min)  # assign task to channel with earliest availability

    def _update_ex(self, n, ch):
        self._ch_ex[n] = ch

        self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
        self._l_ex += self._tasks[n](self._t_ex[n])  # add task execution loss

        self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration  # new channel availability

    def _extend_util(self, seq_ext, inplace=True):
        node = self
        if not inplace:
            node = deepcopy(node)

        node.seq_extend(seq_ext, check_valid=False)

        return node

    def branch(self, do_permute=True, rng=None):
        """
        Generate descendant nodes.

        Parameters
        ----------
        do_permute : bool
            Enables random permutation of returned node list.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Yields
        -------
        TreeNode
            Descendant node with one additional task scheduled.

        """

        seq_iter = list(self._seq_rem)
        if do_permute:
            rng = self._get_rng(rng)
            rng.shuffle(seq_iter)

        for n in seq_iter:
            yield self._extend_util(n, inplace=False)

    def roll_out(self, inplace=True, rng=None):
        """
        Generates/updates node with a randomly completed sequence.

        Parameters
        ----------
        inplace : bool, optional
            Update node in-place or return a new TreeNode object.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        TreeNode
            Only if `inplace` is False.

        """

        rng = self._get_rng(rng)
        seq_ext = rng.permutation(list(self._seq_rem)).tolist()

        node = self._extend_util(seq_ext, inplace)

        if not inplace:
            return node

    def check_swaps(self):
        """Try adjacent task swapping, overwrite node if loss drops."""

        if len(self._seq_rem) != 0:
            raise ValueError("Node sequence must be complete.")

        for i in range(len(self.seq) - 1):
            seq_swap = self.seq.copy()
            seq_swap[i:i + 2] = seq_swap[i:i + 2][::-1]
            node_swap = self.__class__(self._tasks, self._ch_avail, seq_swap)
            if node_swap.l_ex < self.l_ex:
                self = node_swap  # TODO: improper?

    def _earliest_sorter(self, name, inplace=True, check_swaps=False):
        _dict = {n: getattr(self.tasks[n], name) for n in self.seq_rem}
        seq_ext = sorted(self.seq_rem, key=_dict.__getitem__)

        node = self._extend_util(seq_ext, inplace)

        if check_swaps:
            node.check_swaps()

        if not inplace:
            return node

    def earliest_release(self, inplace=True, check_swaps=False):
        return self._earliest_sorter('t_release', inplace, check_swaps)

    def earliest_drop(self, inplace=True, check_swaps=False):
        return self._earliest_sorter('t_drop', inplace, check_swaps)

    def mcts(self, n_mc=1, inplace=True, verbose=False, rng=None):

        # TODO: add exploration/exploitation input control.

        # l_up = TreeNodeBound(tasks, ch_avail).l_up
        tree = SearchNode(self.n_tasks, self.seq, rng=self._get_rng(rng))

        node_best, loss_min = None, np.inf
        while tree.n_visits < n_mc:
            if verbose:
                print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')

            seq = tree.simulate()  # roll-out a complete sequence
            node = self.__class__(self.tasks, self.ch_avail, seq)  # evaluate execution times and channels, total loss

            tree.backup(seq, node.l_ex)  # update search tree from leaf sequence to root

            if node.l_ex < loss_min:
                node_best, loss_min = node, node.l_ex

        if inplace:
            self.seq = node_best.seq
        else:
            return node_best


class TreeNodeBound(TreeNode):
    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        self._l_lo = 0.
        self._l_up = float('inf')
        super().__init__(tasks, ch_avail, seq, rng)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    l_lo = property(lambda self: self._l_lo)
    l_up = property(lambda self: self._l_up)

    def seq_extend(self, seq_ext, check_valid=True):
        """
        Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq_ext : int or Sequence of int
            Indices referencing self.tasks.
        check_valid : bool
            Perform check of index sequence validity.

        """

        super().seq_extend(seq_ext, check_valid)
        self._update_bounds()

    def seq_append(self, n, check_valid=True):
        super().seq_append(n, check_valid)
        self._update_bounds()

    def _update_bounds(self):

        if len(self.seq_rem) == 0:
            return

        t_release_max = max(min(self._ch_avail), *(self._tasks[n].t_release for n in self._seq_rem))
        t_ex_max = t_release_max + sum(self._tasks[n].duration for n in self._seq_rem)
        t_ex_max -= min(self._tasks[n].duration for n in self._seq_rem)

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n](max(min(self._ch_avail), self._tasks[n].t_release))
            self._l_up += self._tasks[n](t_ex_max)

    def branch_bound(self, inplace=True, verbose=False, rng=None):
        """
        Complete node sequence optimally using Branch-and-Bound algorithm.

        Parameters
        ----------
        inplace : bool, optional
            If True, self.seq is completed. Otherwise, a new node object is returned.
        verbose : bool, optional
            Enables printing of algorithm state information.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        TreeNodeBound, optional
            Only if `inplace` is False.

        """

        # TODO: different search strategies? pre-sort?

        rng = self._get_rng(rng)

        stack = deque([self])  # initialize stack
        node_best = stack[0].roll_out(inplace=False, rng=rng)  # roll-out initial solution

        # Iterate
        while len(stack) > 0:
            node = stack.pop()  # extract node

            # Branch
            for node_new in node.branch(do_permute=True, rng=rng):
                # Bound
                if node_new.l_lo < node_best.l_ex:  # new node is not dominated
                    if node_new.l_up < node_best.l_ex:
                        node_best = node_new.roll_out(inplace=False, rng=rng)  # roll-out a new best node
                        stack = [s for s in stack if s.l_lo < node_best.l_ex]  # cut dominated nodes

                    stack.append(node_new)  # add new node to stack, LIFO

            if verbose:
                progress = 1 - sum(factorial(len(node.seq_rem)) for node in stack) / factorial(self.n_tasks)
                print(f'Search progress: {progress:.3f}, Loss < {node_best.l_ex:.3f}', end='\r')
                # print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

        if inplace:
            self.seq = node_best.seq
        else:
            return node_best


class TreeNodeShift(TreeNode):
    _tasks: Sequence[ShiftTask]

    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        self.t_origin = 0.
        super().__init__(tasks, ch_avail, seq, rng)

        self.shift_origin()  # performs initial shift when initialized with empty sequence

    def __repr__(self):
        return f"TreeNodeShift(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    def _update_ex(self, n, ch):
        self._ch_ex[n] = ch

        t_ex_rel = max(self._tasks[n].t_release, self._ch_avail[ch])  # relative to time origin
        self._t_ex[n] = self.t_origin + t_ex_rel  # absolute time
        self._l_ex += self._tasks[n](t_ex_rel)  # add task execution loss

        self._ch_avail[ch] = t_ex_rel + self._tasks[n].duration  # relative to time origin

        self.shift_origin()

    def shift_origin(self):
        """Shifts the time origin to the earliest channel availability and invokes shift method of each task,
        adding each incurred loss to the total."""

        ch_avail_min = min(self._ch_avail)
        if ch_avail_min == 0.:
            return

        self.t_origin += ch_avail_min
        self._ch_avail -= ch_avail_min
        for n, task in enumerate(self._tasks):
            loss_inc = task.shift_origin(ch_avail_min)  # re-parameterize task, return any incurred loss
            if n in self._seq_rem:
                self._l_ex += loss_inc  # add loss incurred due to origin shift for any unscheduled tasks


class SearchNode(RandomGeneratorMixin):
    """
    Node object for Monte Carlo tree search.

    Parameters
    ----------
    seq : Sequence of int
        Partial task index sequence.

    Attributes
    ----------
    seq : Sequence of int
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

    def __init__(self, n_tasks, seq=(), l_up=np.inf, rng=None):
        super().__init__(rng)

        self._n_tasks = n_tasks
        self._seq = list(seq)
        self._l_up = l_up   # TODO: use normalized loss to drive explore/exploit

        self._n_visits = 0
        self._l_avg = 0.
        # self._l_min = np.inf  # FIXME: min? ordered statistic?

        self._children = {}
        self._seq_unk = set(range(self.n_tasks)) - set(self._seq)  # set of unexplored task indices

    n_tasks = property(lambda self: self._n_tasks)
    seq = property(lambda self: self._seq)

    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)
    children = property(lambda self: self._children)

    def __repr__(self):
        return f"SearchNode(seq={self._seq}, children={list(self._children.keys())}, " \
               f"visits={self._n_visits}, avg_loss={self._l_avg:.3f})"

    def __getitem__(self, item):
        """
        Access a descendant node.

        Parameters
        ----------
        item : int or Sequence of int
            Index of sequence of indices for recursive child node selection.

        Returns
        -------
        SearchNode

        """

        if isinstance(item, Integral):
            return self._children[item]
        elif isinstance(item, Sequence):
            node = self
            for n in item:
                node = node._children[n]
            return node
        else:
            raise TypeError

    @property
    def weight(self):
        return self._l_avg - 10 / (self._n_visits + 1)  # FIXME: placeholder. Need real metric and user control.

    def select_child(self):
        """
        Select a child node according to exploration/exploitation objective minimization.

        Returns
        -------
        SearchNode

        """

        # TODO: learn selection function with value network? Fast EST based selection?
        # TODO: add epsilon-greedy selector?

        w = {n: node.weight for (n, node) in self._children.items()}  # descendant node weights
        w.update({n: -10 for n in self._seq_unk})  # base weight for unexplored nodes
        # FIXME: value?

        w = dict(self.rng.permutation(list(w.items())))  # permute elements to break ties randomly

        n = int(min(w, key=w.__getitem__))
        if n not in self._children:
            self._children[n] = SearchNode(self.n_tasks, self._seq + [n], self._l_up, self.rng)
            self._seq_unk.remove(n)

        return self._children[n]

    def simulate(self):
        """
        Produce an index sequence from iterative child node selection.

        Returns
        -------
        Sequence of int

        """

        node = self
        while len(node._seq) < self.n_tasks:
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

    def backup(self, seq, loss):
        """
        Updates search attributes for all descendant nodes corresponding to an index sequence.

        Parameters
        ----------
        seq : Sequence of int
            Complete task index sequence.
        loss : float
            Loss of a complete solution descending from the node.

        """

        if len(seq) != self.n_tasks:
            raise ValueError('Sequence must be complete.')

        seq_rem = seq[len(self._seq):]

        node = self
        node.update_stats(loss)
        for n in seq_rem:
            node = node._children[n]
            node.update_stats(loss)

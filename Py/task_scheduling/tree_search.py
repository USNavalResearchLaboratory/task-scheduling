from collections import deque
from copy import deepcopy
from math import factorial
from typing import Sequence
from operator import attrgetter, methodcaller
from itertools import permutations
from time import perf_counter

import numpy as np
import pandas as pd
from sortedcontainers import SortedKeyList

from task_scheduling.tasks import Base as BaseTask, Shift as ShiftTask
from task_scheduling.util.generic import RandomGeneratorMixin


# TODO: modify classes and algorithms to efficiently handle repeated tasks!?


class TreeNode(RandomGeneratorMixin):  # TODO: rename? TaskSeq?
    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        """
        Node object for mapping sequences into optimal execution times and channels.

        Parameters
        ----------
        tasks : Sequence
        ch_avail : Sequence of float
            Channel availabilities
        seq : Sequence of int
            Partial task index sequence.
        rng : int or RandomState or Generator, optional
                NumPy random number generator or seed. Instance RNG if None.

        """

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

        self.seq = seq
        # if len(seq) > 0:
        #     self.seq_extend(seq)

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
        seq = list(seq)
        seq_prev, seq_ext = seq[:len(self._seq)], seq[len(self._seq):]
        if seq_prev == self._seq:  # new sequence is an extension of current sequence
            self.seq_extend(seq_ext)
            # if len(seq_ext) > 0:
            #     self.seq_extend(seq_ext)
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

        if not isinstance(seq_ext, Sequence):
            seq_ext = [seq_ext]
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
        n : int
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

        if not inplace:
            return node

    def branch(self, permute=True, rng=None):
        """
        Generate descendant nodes.

        Parameters
        ----------
        permute : bool
            Enables random permutation of returned node list.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Yields
        -------
        TreeNode
            Descendant node with one additional task scheduled.

        """

        seq_iter = list(self._seq_rem)
        if permute:
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

        return self._extend_util(seq_ext, inplace)
        # node = self._extend_util(seq_ext, inplace)
        # if not inplace:
        #     return node

    # def check_swaps(self):
    #     """Try adjacent task swapping, overwrite node if loss drops."""
    #
    #     if len(self._seq_rem) != 0:
    #         raise ValueError("Node sequence must be complete.")
    #
    #     for i in range(len(self.seq) - 1):
    #         seq_swap = self.seq.copy()
    #         seq_swap[i:i + 2] = seq_swap[i:i + 2][::-1]
    #         node_swap = self.__class__(self._tasks, self._ch_avail, seq_swap)
    #         if node_swap.l_ex < self.l_ex:
    #             self = node_swap

    def _earliest_sorter(self, name, inplace=True):
        _dict = {n: getattr(self.tasks[n], name) for n in self.seq_rem}
        seq_ext = sorted(self.seq_rem, key=_dict.__getitem__)
        # def sort_func(n):
        #     return getattr(self.tasks[n], name)
        # seq_ext = sorted(self.seq_rem, key=sort_func)

        return self._extend_util(seq_ext, inplace)
        # node = self._extend_util(seq_ext, inplace)
        # if not inplace:
        #     return node

    def earliest_release(self, inplace=True):
        return self._earliest_sorter('t_release', inplace)

    def earliest_drop(self, inplace=True):
        return self._earliest_sorter('t_drop', inplace)

    def mcts(self, runtime, c_explore=0., visit_threshold=0, inplace=True, verbose=False, rng=None):
        """
        Monte Carlo tree search.

        Parameters
        ----------
        runtime : float
            Allotted algorithm runtime.
        c_explore : float, optional
            Exploration weight. Higher values prioritize less frequently visited notes.
        visit_threshold : int, optional
            Nodes with up to this number of visits will select children using the `expansion` method.
        inplace : bool, optional
            If True, self.seq is completed. Otherwise, a new node object is returned.
        verbose : bool, optional
            Enables printing of algorithm state information.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        TreeNode, optional
            Only if `inplace` is False.

        """

        t_run = perf_counter()

        rng = self._get_rng(rng)
        bounds = TreeNodeBound(self.tasks, self.ch_avail).bounds
        root = SearchNode(self.n_tasks, bounds, self.seq, c_explore, visit_threshold, rng=rng)

        node_best, loss_best = None, np.inf
        while perf_counter() - t_run < runtime:
            if verbose:
                print(f'Solutions evaluated: {root.n_visits}, Min. Loss: {loss_best}', end='\r')

            leaf_new = root.selection()  # expansion step happens in selection call

            seq_ext = leaf_new.seq[len(self.seq):]
            node = self._extend_util(seq_ext, inplace=False)
            node.roll_out()  # TODO: rollout with policy?
            if node.l_ex < loss_best:
                node_best, loss_best = node, node.l_ex

            # loss = leaf_new.evaluation()
            loss = node.l_ex  # TODO: combine rollout with optional value func?
            leaf_new.backup(loss)

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
        else:
            return node_best

    def mcts_v1(self, runtime, c_explore=1., inplace=True, verbose=False, rng=None):

        t_run = perf_counter()

        rng = self._get_rng(rng)
        tree = SearchNodeV1(self.n_tasks, self.seq, c_explore=c_explore, rng=rng)

        node_best, loss_best = None, np.inf
        while perf_counter() - t_run < runtime:
            if verbose:
                print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_best}', end='\r')

            # FIXME
            # print(np.array([[node.n_visits, node.l_avg, node.weight] for node in tree.children.values()]))

            seq = tree.simulate()  # roll-out a complete sequence

            seq_ext = seq[len(self.seq):]
            node = self._extend_util(seq_ext, inplace=False)
            if node.l_ex < loss_best:
                node_best, loss_best = node, node.l_ex

            tree.backup(seq, node.l_ex)  # update search tree from leaf sequence to root

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
        else:
            return node_best

    # def mcts(self, n_mc=1, c_explore=0., visit_threshold=0, inplace=True, verbose=False, rng=None):
    #     """
    #     Monte Carlo tree search.
    #
    #     Parameters
    #     ----------
    #     n_mc : int, optional
    #         Number of complete sequences evaluated.
    #     c_explore : float, optional
    #         Exploration weight. Higher values prioritize less frequently visited notes.
    #     visit_threshold : int, optional
    #         Nodes with up to this number of visits will select children using the `expansion` method.
    #     inplace : bool, optional
    #         If True, self.seq is completed. Otherwise, a new node object is returned.
    #     verbose : bool, optional
    #         Enables printing of algorithm state information.
    #     rng : int or RandomState or Generator, optional
    #         NumPy random number generator or seed. Instance RNG if None.
    #
    #     Returns
    #     -------
    #     TreeNode, optional
    #         Only if `inplace` is False.
    #
    #     """
    #
    #     rng = self._get_rng(rng)
    #
    #     bounds = TreeNodeBound(self.tasks, self.ch_avail).bounds
    #     root = SearchNode(self.n_tasks, bounds, self.seq, c_explore, visit_threshold, rng=rng)
    #
    #     node_best, loss_best = None, np.inf
    #     while root.n_visits < n_mc:
    #         if verbose:
    #             print(f'Solutions evaluated: {root.n_visits+1}/{n_mc}, Min. Loss: {loss_best}', end='\r')
    #
    #         leaf_new = root.selection()  # expansion step happens in selection call
    #
    #         seq_ext = leaf_new.seq[len(self.seq):]
    #         node = self._extend_util(seq_ext, inplace=False)
    #         node.roll_out()  # TODO: rollout with policy?
    #         if node.l_ex < loss_best:
    #             node_best, loss_best = node, node.l_ex
    #
    #         # loss = leaf_new.evaluation()
    #         loss = node.l_ex  # TODO: combine rollout with optional value func?
    #         leaf_new.backup(loss)
    #
    #     if inplace:
    #         # self.seq = node_best.seq
    #         seq_ext = node_best.seq[len(self.seq):]
    #         self._extend_util(seq_ext)
    #     else:
    #         return node_best
    #
    # def mcts_v1(self, n_mc=1, c_explore=1., inplace=True, verbose=False, rng=None):
    #
    #     rng = self._get_rng(rng)
    #
    #     # l_up = TreeNodeBound(tasks, ch_avail).l_up  # TODO: normalization?
    #     tree = SearchNodeV1(self.n_tasks, self.seq, c_explore=c_explore, rng=rng)
    #
    #     node_best, loss_best = None, np.inf
    #     while tree.n_visits < n_mc:
    #         if verbose:
    #             print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_best}', end='\r')
    #
    #         # FIXME
    #         # print(np.array([[node.n_visits, node.l_avg, node.weight] for node in tree.children.values()]))
    #
    #         seq = tree.simulate()  # roll-out a complete sequence
    #
    #         seq_ext = seq[len(self.seq):]
    #         node = self._extend_util(seq_ext, inplace=False)
    #         if node.l_ex < loss_best:
    #             node_best, loss_best = node, node.l_ex
    #
    #         tree.backup(seq, node.l_ex)  # update search tree from leaf sequence to root
    #
    #     if inplace:
    #         # self.seq = node_best.seq
    #         seq_ext = node_best.seq[len(self.seq):]
    #         self._extend_util(seq_ext)
    #     else:
    #         return node_best

    def brute_force(self, inplace=True, verbose=False):
        """
        Exhaustively search all complete sequences.

        Parameters
        ----------
        inplace : bool, optional
            If True, self.seq is completed. Otherwise, a new node object is returned.
        verbose : bool
            Enables printing of algorithm state information.

        Returns
        -------
        TreeNode
            Only if `inplace` is False.

        """

        node_best, loss_best = None, float('inf')

        n_perms = factorial(len(self.seq_rem))
        for i, seq in enumerate(permutations(self.seq_rem)):
            if verbose:
                print(f"Brute force: {i + 1}/{n_perms}", end='\r')

            node = self._extend_util(seq, inplace=False)
            if node.l_ex < loss_best:
                node_best, loss_best = node, node.l_ex

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
        else:
            return node_best


class TreeNodeBound(TreeNode):
    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        self._bounds = [0., float('inf')]
        super().__init__(tasks, ch_avail, seq, rng)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    bounds = property(lambda self: self._bounds)
    l_lo = property(lambda self: self._bounds[0])
    l_up = property(lambda self: self._bounds[1])

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

        self._bounds = [self._l_ex, self._l_ex]
        if len(self.seq_rem) == 0:
            return  # already converged

        t_release_max = max(min(self._ch_avail), *(self._tasks[n].t_release for n in self._seq_rem))
        t_ex_max = t_release_max + sum(self._tasks[n].duration for n in self._seq_rem)
        # t_ex_max -= min(self._tasks[n].duration for n in self._seq_rem)

        for n in self._seq_rem:  # update loss bounds
            self._bounds[0] += self._tasks[n](max(min(self._ch_avail), self._tasks[n].t_release))
            self._bounds[1] += self._tasks[n](t_ex_max)

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

        rng = self._get_rng(rng)

        node_best = self.roll_out(inplace=False, rng=rng)  # roll-out initial solution
        stack = deque([self])  # initialize stack
        # stack = [self]

        # Iterate
        while len(stack) > 0:
            node = stack.pop()  # extract node
            if node.l_lo >= node_best.l_ex:
                continue  # node is dominated

            # Branch
            for node_new in node.branch(permute=True, rng=rng):
                # Bound
                if node_new.l_lo < node_best.l_ex:
                    stack.append(node_new)  # new node is not dominated, add to stack (LIFO)

                    if node_new.l_up < node_best.l_ex:
                        node_best = node_new.roll_out(inplace=False, rng=rng)  # roll-out a new best node

            # stack.sort(key=attrgetter('l_lo'), reverse=True)

            if verbose:
                progress = 1 - sum(factorial(len(node.seq_rem)) for node in stack) / factorial(self.n_tasks)
                print(f'Search progress: {progress:.3f}, Loss < {node_best.l_ex:.3f}', end='\r')
                # print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
        else:
            return node_best

    def branch_bound_priority(self, priority_func=None, heuristic=None, inplace=True, verbose=False):
        """
        Branch-and-Bound with priority queueing and variable heuristics.

        Parameters
        ----------
        priority_func : callable, optional
            Key function that maps `TreeNode` objects to priority values. Defaults to negative lower bound.
        heuristic : callable, optional
            Uses a partial node to generate a complete sequence node.
        inplace : bool, optional
            If True, self.seq is completed. Otherwise, a new node object is returned.
        verbose : bool, optional
            Enables printing of algorithm state information.

        Returns
        -------
        TreeNodeBound, optional
            Only if `inplace` is False.

        """

        if priority_func is None:
            def priority_func(node_):
                return -node_.l_lo

        if heuristic is None:
            heuristic = methodcaller('roll_out', inplace=False)
            # heuristic = methodcaller('earliest_release', inplace=False)

        node_best = heuristic(self)
        stack = SortedKeyList([self], priority_func)

        # Iterate
        while len(stack) > 0:
            node = stack.pop()  # extract node
            if node.l_lo >= node_best.l_ex:
                continue  # node is dominated

            # Branch
            for node_new in node.branch():
                # Bound
                if node_new.l_lo < node_best.l_ex:
                    stack.add(node_new)  # new node is not dominated, add to stack (prioritized)

                    if node_new.l_up < node_best.l_ex:
                        node_best = heuristic(node_new)

            if verbose:
                progress = 1 - sum(factorial(len(node.seq_rem)) for node in stack) / factorial(self.n_tasks)
                print(f'Search progress: {progress:.3f}, Loss < {node_best.l_ex:.3f}', end='\r')
                # print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
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
    def __init__(self, n_tasks, bounds, seq=(), c_explore=0., visit_threshold=0, parent=None, rng=None):
        """
        Node object for Monte Carlo Tree Search.

        Parameters
        ----------
        n_tasks : int
        bounds : Sequence of float
            Lower and upper loss bounds for node value normalization
        seq : Sequence of int
            Partial task index sequence.
        c_explore : float, optional
            Exploration weight. Higher values prioritize searching new branches.
        visit_threshold : int, optional
            Once node has been visited this many times, UCT is used for child selection, not random choice.
        parent : SearchNode, optional
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        """

        super().__init__(rng)

        self._n_tasks = n_tasks
        self._bounds = tuple(bounds)
        self._seq = list(seq)

        self._c_explore = c_explore
        self._visit_threshold = visit_threshold

        self._parent = parent
        self._children = {}
        self._seq_rem = set(range(self.n_tasks)) - set(self._seq)

        self._n_visits = 0
        self._l_avg = 0.  # TODO: try using min? ordered statistic?

    n_tasks = property(lambda self: self._n_tasks)
    seq = property(lambda self: self._seq)

    parent = property(lambda self: self._parent)
    children = property(lambda self: self._children)

    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)

    def __repr__(self):
        return f"SearchNode(seq={self._seq}, children={list(self._children.keys())}, " \
               f"visits={self._n_visits}, avg_loss={self._l_avg:.3f})"

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return len(self._children) == 0

    @property
    def weight(self):
        """Weight for child selection. Combines average loss with a visit count bonus."""

        value_loss = (self._bounds[1] - self._l_avg) / (self._bounds[1] - self._bounds[0])  # TODO: redundant eval!
        value_explore = np.sqrt(np.log(self.parent.n_visits) / self._n_visits)
        # value_explore = np.sqrt(self.parent.n_visits) / (self._n_visits + 1)
        return value_loss + self._c_explore * value_explore

    def select_child(self):
        """Select child node. Under the threshold, the expansion method is used. Above, the weight property is used."""

        if self._n_visits <= self._visit_threshold:
            return self.expansion()
        else:
            w = {n: child.weight for (n, child) in self._children.items()}  # descendant node weights
            n = max(w, key=w.__getitem__)
            return self.children[n]

    def selection(self):
        """Iteratively select descendant nodes until a leaf is reached."""

        node = self
        while not node.is_leaf:
            node = node.select_child()
        if node.n_visits > 0 and len(node.seq) < node.n_tasks:  # node is not new, expand to create child
            node = node.expansion()

        return node

    def _add_child(self, n):
        self._children[n] = self.__class__(self.n_tasks, self._bounds, self._seq + [n], self._c_explore,
                                           self._visit_threshold, parent=self, rng=self.rng)

    def expansion(self):
        """Pseudo-random expansion, potentially creating a new child node."""

        n = self.rng.choice(list(self._seq_rem))  # TODO: pseudo-random strategies? ERT?
        if n not in self._children:
            self._add_child(n)
        return self._children[n]

    def evaluation(self):  # TODO: user custom definition. EST or NN!
        raise NotImplementedError

    def backup(self, loss):
        """Update loss and visit statistics for the node and all ancestors."""

        node = self
        node.update_stats(loss)
        while not node.is_root:
            node = node.parent
            node.update_stats(loss)

    def update_stats(self, loss):
        """
        Update visit count and average loss evaluated sequences.

        Parameters
        ----------
        loss : float
            Loss of a complete solution descending from the node.

        """

        loss_total = self._l_avg * self._n_visits + loss
        self._n_visits += 1
        self._l_avg = loss_total / self._n_visits


class SearchNodeV1(RandomGeneratorMixin):
    def __init__(self, n_tasks, seq=(), parent=None, c_explore=1., l_up=np.inf, rng=None):
        super().__init__(rng)

        self._n_tasks = n_tasks
        self._seq = list(seq)

        self._parent = parent
        self._children = {}
        self._seq_unk = set(range(self.n_tasks)) - set(self._seq)  # set of unexplored task indices

        self._c_explore = c_explore
        self._l_up = l_up   # TODO: use normalized loss to drive explore/exploit

        self._n_visits = 0
        self._l_avg = 0.  # TODO: try using min? ordered statistic?

    n_tasks = property(lambda self: self._n_tasks)
    seq = property(lambda self: self._seq)

    parent = property(lambda self: self._parent)
    children = property(lambda self: self._children)

    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)

    def __repr__(self):
        return f"SearchNodeV1(seq={self._seq}, children={list(self._children.keys())}, " \
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
        SearchNodeV1

        """

        if isinstance(item, int):
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
        # return self._l_avg - self._c_explore / (self._n_visits + 1)
        # return self._l_avg - self._c_explore * np.sqrt(self.parent.n_visits) / (self._n_visits + 1)
        return self._l_avg - self._c_explore * np.sqrt(np.log(self.parent.n_visits) / self._n_visits)

    def select_child(self):
        """
        Select a child node according to exploration/exploitation objective minimization.

        Returns
        -------
        SearchNodeV1

        """

        # TODO: learn selection function with value network? Fast EST based selection?
        # TODO: add epsilon-greedy selector?

        w = {n: node.weight for (n, node) in self._children.items()}  # descendant node weights
        w.update({n: -self._c_explore for n in self._seq_unk})  # base weight for unexplored nodes
        # FIXME: init value? use upper bound??

        w = dict(self.rng.permutation(list(w.items())))  # permute elements to break ties randomly

        n = int(min(w, key=w.__getitem__))
        if n not in self._children:
            self._children[n] = self.__class__(self.n_tasks, self._seq + [n], parent=self, c_explore=self._c_explore,
                                               l_up=self._l_up, rng=self.rng)
            self._seq_unk.remove(n)

        return self._children[n]

    def simulate(self):
        """
        Produce an index sequence from iterative child node selection.

        Returns
        -------
        list of int

        """

        node = self
        while len(node._seq) < self.n_tasks:
            node = node.select_child()
        return node._seq

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

        seq_rem = seq[len(self._seq):]  # TODO: check leading values for equivalence?

        node = self
        node.update_stats(loss)
        for n in seq_rem:
            node = node._children[n]
            node.update_stats(loss)

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

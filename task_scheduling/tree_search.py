from collections import deque
from copy import deepcopy
from math import factorial
from typing import Sequence
from operator import methodcaller
from itertools import permutations
from time import perf_counter

import numpy as np
import pandas as pd
from sortedcontainers import SortedKeyList

from task_scheduling.base import RandomGeneratorMixin
from task_scheduling.tasks import Shift as ShiftTask

# TODO: make problem a shared class attribute? Make a class constructor?


class ScheduleNode(RandomGeneratorMixin):
    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        """
        Node object for mapping task sequences into execution schedules.

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

        self._tasks = list(tasks)
        # self._tasks = deepcopy(tasks)
        self._ch_avail = np.array(ch_avail, dtype=float)

        if min(self._ch_avail) < 0.:
            raise ValueError("Initial channel availabilities must be non-negative.")

        self._seq = []
        self._seq_rem = set(range(self.n_tasks))

        self._t_ex = np.full(self.n_tasks, np.nan)
        self._ch_ex = np.full(self.n_tasks, -1)

        self._l_ex = 0.  # incurred loss

        self.seq = seq

    def __repr__(self):
        return f"ScheduleNode(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    def __eq__(self, other):
        if isinstance(other, ScheduleNode):
            return (self.tasks, self.ch_avail, self.seq) == (other.tasks, other.ch_avail, other.seq)
        else:
            return NotImplemented

    def summary(self, file=None):
        """Print a string describing important node attributes."""
        keys = ('seq', 't_ex', 'ch_ex', 'l_ex')
        df = pd.Series({key: getattr(self, key) for key in keys})
        print(df.to_markdown(tablefmt='github', floatfmt='.3f'), file=file)

        # str_out = f'ScheduleNode\n- sequence: {self.seq}\n- execution times: {self.t_ex}' \
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
        else:
            # self.__init__(self.tasks, self.ch_avail, seq, rng=self.rng)  # initialize from scratch
            raise ValueError(f"Sequence must be an extension of {self._seq}")  # shift nodes cannot recover tasks

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
        ScheduleNode
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
            Update node in-place or return a new ScheduleNode object.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Returns
        -------
        ScheduleNode
            Only if `inplace` is False.

        """

        rng = self._get_rng(rng)
        seq_ext = rng.permutation(list(self._seq_rem)).tolist()

        return self._extend_util(seq_ext, inplace)

    def _earliest_sorter(self, name, inplace=True):
        _dict = {n: getattr(self.tasks[n], name) for n in self.seq_rem}
        seq_ext = sorted(self.seq_rem, key=_dict.__getitem__)

        return self._extend_util(seq_ext, inplace)

    def earliest_release(self, inplace=True):
        return self._earliest_sorter('t_release', inplace)

    def earliest_drop(self, inplace=True):
        return self._earliest_sorter('t_drop', inplace)

    def mcts(self, max_runtime=np.inf, max_rollouts=None, c_explore=0., visit_threshold=0, inplace=True, verbose=False,
             rng=None):
        """
        Monte Carlo tree search.

        Parameters
        ----------
        max_runtime : float, optional
            Allotted algorithm runtime.
        max_rollouts : int, optional
            Maximum number of rollouts allowed.
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
        ScheduleNode, optional
            Only if `inplace` is False.

        Notes
        -----
        Reproducibility using `rng` argument may not be guaranteed due to differing code execution times.

        """

        t_run = perf_counter()

        if max_rollouts is None:
            max_rollouts = np.inf

        rng = self._get_rng(rng)
        bounds = ScheduleNodeBound(self.tasks, self.ch_avail).bounds
        root = MCTSNode(self.n_tasks, bounds, self.seq, c_explore, visit_threshold, rng=rng)

        node_best, loss_best = None, np.inf
        while True:
            if verbose:
                print(f'# rollouts: {root.n_visits}, Min. Loss: {loss_best}', end='\r')

            leaf_new = root.selection()  # expansion step happens in selection call

            seq_ext = leaf_new.seq[len(self.seq):]
            node = self._extend_util(seq_ext, inplace=False)
            node.roll_out(rng=rng)  # TODO: rollout with learned policy?
            if node.l_ex < loss_best:
                node_best, loss_best = node, node.l_ex

            # loss = leaf_new.evaluation()
            loss = node.l_ex  # TODO: mix rollout loss with value func, like AlphaGo?
            leaf_new.backup(loss)

            if perf_counter() - t_run >= max_runtime or root.n_visits >= max_rollouts:
                break

        # if verbose:
        #     print(f"Total # rollouts: {root.n_visits}, loss={loss_best}")

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq):]
            self._extend_util(seq_ext)
        else:
            return node_best

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
        ScheduleNode
            Only if `inplace` is False.

        """

        node_best, loss_best = None, np.inf

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


class ScheduleNodeBound(ScheduleNode):
    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        self._bounds = [0., np.inf]
        super().__init__(tasks, ch_avail, seq, rng)

    def __repr__(self):
        return f"ScheduleNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

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
        ScheduleNodeBound, optional
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
        Branch-and-Bound with priority queueing and user-defined heuristics.

        Parameters
        ----------
        priority_func : callable, optional
            Key function that maps `ScheduleNode` objects to priority values. Defaults to negative lower bound.
        heuristic : callable, optional
            Uses a partial node to generate a complete sequence node.
        inplace : bool, optional
            If True, self.seq is completed. Otherwise, a new node object is returned.
        verbose : bool, optional
            Enables printing of algorithm state information.

        Returns
        -------
        ScheduleNodeBound, optional
            Only if `inplace` is False.

        """

        if priority_func is None:
            def priority_func(node_):
                return -node_.l_lo

        if heuristic is None:
            heuristic = methodcaller('roll_out', inplace=False)

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


class ScheduleNodeShift(ScheduleNode):
    _tasks: Sequence[ShiftTask]

    def __init__(self, tasks, ch_avail, seq=(), rng=None):
        self.t_origin = 0.
        tasks = deepcopy(tasks)  # tasks modified in-place during `shift_origin`
        super().__init__(tasks, ch_avail, seq, rng)

        self.shift_origin()  # performs initial shift when initialized with empty sequence

    def __repr__(self):
        return f"ScheduleNodeShift(sequence: {self.seq}, loss incurred:{self.l_ex:.3f})"

    # def _update_ex(self, n, ch):
    #     self._ch_ex[n] = ch
    #
    #     t_ex_rel = max(self._tasks[n].t_release, self._ch_avail[ch])  # relative to time origin
    #     self._t_ex[n] = self.t_origin + t_ex_rel  # absolute time
    #     self._l_ex += self._tasks[n](t_ex_rel)  # add task execution loss
    #
    #     self._ch_avail[ch] = t_ex_rel + self._tasks[n].duration  # relative to time origin
    #
    #     self.shift_origin()

    def _update_ex(self, n, ch):
        super()._update_ex(n, ch)
        self._t_ex[n] += self.t_origin  # relative to absolute
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


class MCTSNode(RandomGeneratorMixin):
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
        parent : MCTSNode, optional
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
    seq_rem = property(lambda self: self._seq_rem)

    parent = property(lambda self: self._parent)
    children = property(lambda self: self._children)

    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)

    def __repr__(self):
        return f"MCTSNode(seq={self._seq}, children={list(self._children.keys())}, " \
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

        # TODO: use parent policy eval to influence weighting

        value_loss = (self._bounds[1] - self._l_avg) / (self._bounds[1] - self._bounds[0])
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
        if node.n_visits > 0 and len(node.seq_rem) > 0:  # node is not new, expand to create child
            node = node.expansion()

        return node

    def _add_child(self, n):
        self._children[n] = self.__class__(self.n_tasks, self._bounds, self._seq + [n], self._c_explore,
                                           self._visit_threshold, parent=self, rng=self.rng)

    def expansion(self):
        """Pseudo-random expansion, potentially creating a new child node."""

        n = self.rng.choice(list(self._seq_rem))  # TODO: pseudo-random w/ policy?
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

from typing import Sequence

import numpy as np

from task_scheduling._core import RandomGeneratorMixin
from task_scheduling.tree_search import TreeNode


class MCTSv1Mixin:
    def mcts_v1(self, runtime, c_explore=1.0, inplace=True, verbose=False, rng=None):

        t_run = perf_counter()

        rng = self._get_rng(rng)
        tree = SearchNodeV1(self.n_tasks, self.seq, c_explore=c_explore, rng=rng)

        node_best, loss_best = None, np.inf
        while perf_counter() - t_run < runtime:
            if verbose:
                print(
                    f"Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_best}",
                    end="\r",
                )

            seq = tree.simulate()  # roll-out a complete sequence

            seq_ext = seq[len(self.seq) :]
            node = self._extend_util(seq_ext, inplace=False)
            if node.l_ex < loss_best:
                node_best, loss_best = node, node.l_ex

            tree.backup(seq, node.l_ex)  # update search tree from leaf sequence to root

        if inplace:
            # self.seq = node_best.seq
            seq_ext = node_best.seq[len(self.seq) :]
            self._extend_util(seq_ext)
        else:
            return node_best


class TreeNodeMCTSv1(MCTSv1Mixin, TreeNode):
    pass


class SearchNodeV1(RandomGeneratorMixin):
    def __init__(
        self, n_tasks, seq=(), parent=None, c_explore=1.0, l_up=np.inf, rng=None
    ):
        super().__init__(rng)

        self._n_tasks = n_tasks
        self._seq = list(seq)

        self._parent = parent
        self._children = {}
        self._seq_unk = set(range(self.n_tasks)) - set(
            self._seq
        )  # set of unexplored task indices

        self._c_explore = c_explore
        self._l_up = l_up

        self._n_visits = 0
        self._l_avg = 0.0

    n_tasks = property(lambda self: self._n_tasks)
    seq = property(lambda self: self._seq)

    parent = property(lambda self: self._parent)
    children = property(lambda self: self._children)

    n_visits = property(lambda self: self._n_visits)
    l_avg = property(lambda self: self._l_avg)

    def __repr__(self):
        return (
            f"SearchNodeV1(seq={self._seq}, children={list(self._children.keys())}, "
            f"visits={self._n_visits}, avg_loss={self._l_avg:.3f})"
        )

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
        return self._l_avg - self._c_explore * np.sqrt(
            np.log(self.parent.n_visits) / self._n_visits
        )

    def select_child(self):
        """
        Select a child node according to exploration/exploitation objective minimization.

        Returns
        -------
        SearchNodeV1

        """

        w = {
            n: node.weight for (n, node) in self._children.items()
        }  # descendant node weights
        w.update(
            {n: -self._c_explore for n in self._seq_unk}
        )  # base weight for unexplored nodes
        # FIXME: init value? use upper bound??

        w = dict(
            self.rng.permutation(list(w.items()))
        )  # permute elements to break ties randomly

        n = int(min(w, key=w.__getitem__))
        if n not in self._children:
            self._children[n] = self.__class__(
                self.n_tasks,
                self._seq + [n],
                parent=self,
                c_explore=self._c_explore,
                l_up=self._l_up,
                rng=self.rng,
            )
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
            raise ValueError("Sequence must be complete.")

        seq_prev, seq_rem = seq[: len(self._seq)], seq[len(self._seq) :]
        if seq_prev != self._seq:
            raise ValueError(f"Sequence must be an extension of {self._seq}")

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


def mcts_v1(tasks, ch_avail, runtime, c_explore=1.0, verbose=False, rng=None):
    """
    Monte Carlo tree search algorithm.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
    ch_avail : Sequence of float
        Channel availability times.
    runtime : float
            Allotted algorithm runtime.
    c_explore : float, optional
        Exploration weight. Higher values prioritize unexplored tree nodes.
    verbose : bool
        Enables printing of algorithm state information.
    rng : int or RandomState or Generator, optional
        NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    node = TreeNodeMCTSv1(tasks, ch_avail, rng=rng)
    node = node.mcts_v1(runtime, c_explore, inplace=False, verbose=verbose)

    return node.t_ex, node.ch_ex

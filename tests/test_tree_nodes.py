from math import isclose

import numpy as np

from task_scheduling.algorithms import branch_bound
from task_scheduling.tree_search import ScheduleNode, ScheduleNodeShift
from task_scheduling.util import evaluate_schedule
from task_scheduling.generators import problems as problem_gens


def test_argsort():
    """Check that seq=np.argsort(sch['t']) maps to an optimal schedule."""

    problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1)
    for i in range(10):
        print(f"{i}", end='\n')

        (tasks, ch_avail), = problem_gen(1)
        sch = branch_bound(tasks, ch_avail, verbose=True, rng=None)
        loss = evaluate_schedule(tasks, sch)

        seq_sort = np.argsort(sch['t'])
        node_sort = ScheduleNode(tasks, ch_avail, seq_sort)

        assert isclose(loss, node_sort.loss)


def test_shift():
    """Check accuracy of ScheduleNodeShift solution."""

    problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1)
    for i in range(10):
        print(f"{i}", end='\n')

        (tasks, ch_avail), = problem_gen(1)
        seq = np.random.permutation(problem_gen.n_tasks)
        node, node_s = ScheduleNode(tasks, ch_avail, seq), ScheduleNodeShift(tasks, ch_avail, seq)
        assert np.allclose(node.sch['t'], node_s.sch['t'])
        assert abs(node.loss - node_s.loss) < 1e-9


def main():
    test_argsort()
    test_shift()


if __name__ == '__main__':
    main()

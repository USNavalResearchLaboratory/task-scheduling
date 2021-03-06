from math import isclose

import numpy as np

from task_scheduling.algorithms import branch_bound
from task_scheduling.generators import problems as problem_gens
from task_scheduling.nodes import ScheduleNode, ScheduleNodeReform
from task_scheduling.util import evaluate_schedule

n_ch = 2
n_tasks = 8

ch_avail_lim = (-1, 1)

problem_gen_list = [
    problem_gens.Random.continuous_linear_drop(
        n_tasks,
        n_ch,
        ch_avail_lim,
        duration_lim=(0.03, 0.06),
        t_release_lim=(-4, 4),
        slope_lim=(0.5, 2),
        t_drop_lim=(6, 12),
        l_drop_lim=(35, 50),
    ),
    problem_gens.Random.continuous_exp(
        n_tasks, n_ch, ch_avail_lim, duration_lim=(0.03, 0.06), t_release_lim=(-4, 4)
    ),
]


def test_argsort():
    """Check that seq=np.argsort(sch['t']) maps to an optimal schedule."""
    for problem_gen in problem_gen_list:
        for i, (tasks, ch_avail) in enumerate(problem_gen(10)):
            print(f"{i}", end="\n")

            sch = branch_bound(tasks, ch_avail, verbose=True, rng=None)
            loss = evaluate_schedule(tasks, sch)

            seq_sort = np.argsort(sch["t"])
            node_sort = ScheduleNode(tasks, ch_avail, seq_sort)

            assert isclose(loss, node_sort.loss)


def test_reform():
    """Check accuracy of ScheduleNodeReform solution."""
    for problem_gen in problem_gen_list:
        for i, (tasks, ch_avail) in enumerate(problem_gen(1000)):
            print(f"{i}", end="\n")

            seq = np.random.permutation(problem_gen.n_tasks)
            node = ScheduleNode(tasks, ch_avail, seq)
            node_s = ScheduleNodeReform(tasks, ch_avail, seq)

            assert np.allclose(node.sch["t"], node_s.sch["t"])
            assert np.isclose(node.loss, node_s.loss)


if __name__ == "__main__":
    test_argsort()
    test_reform()

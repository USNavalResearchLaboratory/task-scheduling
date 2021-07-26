from math import isclose

import numpy as np

from task_scheduling.tree_search import TreeNode, TreeNodeShift, TreeNodeBound
from task_scheduling.algorithms.free import branch_bound
from task_scheduling.util.results import evaluate_schedule
from task_scheduling.generators import scheduling_problems as problem_gens, tasks as task_gens

# TODO: use proper testing w/ builtin module

problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1)


# def test_argsort(problem_gen, n_iter=1):
def test_argsort():
    """check that seq=np.argsort(t_ex) maps to an optimal schedule"""
    n_iter = 10
    for i in range(n_iter):
        print(f"{i}", end='\n')

        # seq = np.random.permutation(n_tasks)
        # node = TreeNode(seq)
        # t_ex = node.t_ex

        (tasks, ch_avail), = problem_gen(1)
        t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True, rng=None)
        loss = evaluate_schedule(tasks, t_ex)

        seq_sort = np.argsort(t_ex)
        node_sort = TreeNode(tasks, ch_avail, seq_sort)
        # t_ex_sort = node_sort.t_ex

        assert isclose(loss, node_sort.l_ex)
        # assert t_ex.tolist() == t_ex_sort.tolist()
        # assert seq.tolist() == seq_sort.tolist()


# def test_shift(problem_gen, n_iter=1):
def test_shift():
    """check accuracy of TreeNodeShift solution"""
    n_iter = 10
    for i in range(n_iter):
        print(f"{i}", end='\n')

        (tasks, ch_avail), = problem_gen(1)
        seq = np.random.permutation(problem_gen.n_tasks)
        node, node_s = TreeNode(tasks, ch_avail, seq), TreeNodeShift(tasks, ch_avail, seq)
        # print(node.t_ex)
        # print(node_s.t_ex)
        assert np.allclose(node.t_ex, node_s.t_ex)
        assert abs(node.l_ex - node_s.l_ex) < 1e-9


def main():
    # tasks = list(task_gens.ContinuousUniformIID.relu_drop()(4))
    # # node = TreeNodeBound(tasks, ch_avail=[0, 0], seq=[2, 0])
    # branch_bound(tasks, ch_avail)

    # problem_gen = problem_gens.Random.continuous_relu_drop(n_tasks=8, n_ch=1)
    # test_argsort(problem_gen, n_iter=10)
    # test_shift(problem_gen, n_iter=10)
    test_argsort()
    test_shift()


if __name__ == '__main__':
    main()

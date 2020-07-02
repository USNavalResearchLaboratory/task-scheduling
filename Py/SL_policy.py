import numpy as np
import matplotlib.pyplot as plt

from util.generic import algorithm_repr, check_rng
from util.results import check_valid, eval_loss

from tasks import ReluDropGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release


def obs_relu_drop(tasks):
    """Generate observation array from list of tasks."""

    # _params = [(task.duration, task.t_release, task.slope, task.t_drop, task.l_drop) for task in tasks]
    # params = np.array(_params, dtype=[('duration', np.float), ('t_release', np.float),
    #                                   ('slope', np.float), ('t_drop', np.float), ('l_drop', np.float)])
    # params.view(np.float).reshape(*params.shape, -1)
    return np.asarray([[task.duration, task.t_release, task.slope, task.t_drop, task.l_drop] for task in tasks])


def data_gen(n_gen, n_tasks, task_gen, n_channels, ch_avail_gen):
    for _ in range(n_gen):
        tasks = task_gen.rand_tasks(n_tasks)
        ch_avail = ch_avail_gen(n_channels)

        t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=False)

        # check_valid(tasks, t_ex, ch_ex)
        # l_ex = eval_loss(tasks, t_ex)

        seq = np.argsort(t_ex).tolist()     # optimal sequence
        state = np.concatenate((np.ones((n_tasks, 1)), obs_relu_drop(tasks)), axis=1)

        # Y = np.zeros((n_tasks-1, n_tasks))
        for n in seq[:-1]:
            x = state
            y = np.zeros(n_tasks)
            y[n] = 1
            state[n, 0] = 0


def main():
    n_gen = 2  # number of task scheduling problems

    n_tasks = 8
    n_channels = 2

    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        # TODO: rng is a mutable default argument!
        return rng.uniform(0, 2, n_ch)

    data_gen(n_gen, n_tasks, task_gen, n_channels, ch_avail_gen)


if __name__ == '__main__':
    main()

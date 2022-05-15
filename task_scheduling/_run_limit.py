from functools import wraps
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

from task_scheduling.nodes import ScheduleNodeBound
from task_scheduling.util import (
    check_schedule,
    evaluate_schedule,
    plot_schedule,
    plot_task_losses,
)


def branch_bound(tasks: list, ch_avail: list, runtimes: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : Collection of tasks.Base
    ch_avail : Collection of float
        Channel availability times.
    runtimes : Collection of float
        Allotted algorithm runtime.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    numpy.ndarray
        Task execution times/channels.

    """

    t_run = perf_counter()

    stack = [ScheduleNodeBound(tasks, ch_avail, rng=rng)]  # initialize stack
    node_best = stack[0].roll_out(inplace=False)  # roll-out initial solution

    # Iterate
    i_time = 0
    n_times = len(runtimes)
    while i_time < n_times and len(stack) > 0:
        node = stack.pop()  # extract node

        # Branch
        for node_new in node.branch(permute=True):
            # Bound
            if node_new.l_lo < node_best.loss:  # new node is not dominated
                if node_new.l_up < node_best.loss:
                    node_best = node_new.roll_out(
                        inplace=False
                    )  # roll-out a new best node
                    stack = [
                        s for s in stack if s.l_lo < node_best.loss
                    ]  # cut dominated nodes

                stack.append(node_new)  # add new node to stack, LIFO

            # Check run conditions
            if perf_counter() - t_run >= runtimes[i_time]:
                yield node_best.sch
                i_time += 1
                if i_time == n_times:
                    break

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.loss:.3f}', end='\r')
            print(
                f"# Remaining Nodes = {len(stack)}, Loss <= {node_best.loss:.3f}",
                end="\r",
            )

    for _ in range(i_time, n_times):
        yield node_best.sch


# MCTS WIP

# def mcts(tasks, ch_avail, runtimes, c_explore=0., visit_threshold=0, verbose=False, rng=None):
#
#     # node = ScheduleNode(tasks, ch_avail, rng=rng)
#     # node = node.mcts(runtimes, c_explore, visit_threshold, inplace=False, verbose=verbose)
#     # return node.sch
#     raise NotImplementedError


# def mcts_v1(tasks, ch_avail, runtimes, c_explore=1., verbose=False, rng=None):
#
#     # node = ScheduleNode(tasks, ch_avail, rng=rng)
#     # node = node.mcts_v1(runtimes, c_explore, inplace=False, verbose=verbose)
#     # return node.sch
#     raise NotImplementedError


# def mcts(tasks: list, ch_avail: list, runtimes: list, verbose=False):
#
#     # TODO: add early termination for completed search.
#
#     t_run = perf_counter()
#     if not isinstance(runtimes, (Sequence, np.ndarray)):
#         runtimes = [runtimes]
#     if inplace:
#         runtimes = runtimes[-1:]
#     i_time = 0
#     n_times = len(runtimes)
#
#     l_up = ScheduleNodeBound(tasks, ch_avail).l_up
#     tree = SearchNodeV1(n_tasks=len(tasks), seq=[], l_up=l_up)
#
#     node_best = None
#     loss_min = float('inf')
#
#     i_time = 0
#     n_times = len(runtimes)
#     while i_time < n_times:
#         if verbose:
#             print(f'Solutions evaluated: {tree.n_visits}, Min. Loss: {loss_min}', end='\r')
#
#         seq = tree.simulate()  # roll-out a complete sequence
#         node = ScheduleNode(tasks, ch_avail, seq)  # evaluate execution times and channels, total loss
#
#         loss = node.loss
#         tree.backup(seq, loss)  # update search tree from leaf sequence to root
#
#         if loss < loss_min:
#             node_best, loss_min = node, loss
#
#         if perf_counter() - t_run >= runtimes[i_time]:
#             yield node_best.sch
#             i_time += 1

# def mcts_orig(tasks: list, ch_avail: list, max_runtime=float('inf'), n_mc=None, verbose=False, rng=None):
#
#     t_run = perf_counter()
#     run = True
#
#     node = ScheduleNode(tasks, ch_avail, rng=rng)
#     node_best = node.roll_out(do_copy=True)
#
#     n_tasks = len(tasks)
#     if n_mc is None:
#         n_mc = [floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)]
#     elif type(n_mc) == int:
#         n_mc = n_tasks * [n_mc]
#
#     def _roll_outs(n_roll):
#         nonlocal node_best, run
#
#         for _ in range(n_roll):
#             node_mc = node.roll_out(do_copy=True)
#
#             if node_mc.loss < node_best.loss:  # Update best node
#                 node_best = node_mc
#
#             # Check run conditions
#             if perf_counter() - t_run >= max_runtime:
#                 run = False
#                 return
#
#     for n in range(n_tasks):
#         if verbose:
#             print(f'Assigning Task {n + 1}/{n_tasks}', end='\r')
#
#         _roll_outs(n_mc[n])
#
#         if not run:
#             break
#
#         # Assign next task from earliest available channel
#         node.seq_extend(node_best.seq[n], check_valid=False)
#
#     return node_best.sch


def plot_loss_runtime(t_run, loss, do_std=False, ax=None, ax_kwargs=None):
    """
    Line plot of total execution loss versus maximum runtime.

    Parameters
    ----------
    t_run : numpy.ndarray
        Runtime of algorithm.
    loss : numpy.ndarray
        Total loss of scheduled tasks.
    do_std : bool
        Activates error bars for sample standard deviation.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """

    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    names = loss.dtype.names
    for i_name, name in enumerate(names):
        l_mean = loss[name].mean(-1)
        ax.plot(t_run, l_mean, label=name)
        if do_std:
            l_std = loss[name].std(-1)
            # ax.errorbar(t_run, l_mean, yerr=l_std, label=name, errorevery=(i_name, len(names)))
            ax.fill_between(t_run, l_mean - l_std, l_mean + l_std, alpha=0.25)
        # else:
        #     ax.plot(t_run, l_mean, label=name)

    ax.set(xlabel="Runtime (s)", ylabel="Loss")
    ax.legend()
    ax.set(**ax_kwargs)


# def timing_wrapper(scheduler):  # TODO: delete?
#     """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""
#
#     @wraps(scheduler)
#     def timed_scheduler(tasks, ch_avail):
#         t_start = perf_counter()
#         # t_ex, ch_ex = scheduler(tasks, ch_avail)
#         # t_run = perf_counter() - t_start
#         # # return t_ex, ch_ex, t_run
#
#         sch = scheduler(tasks, ch_avail)
#         t_run = perf_counter() - t_start
#
#         # return SchedulingSolution(*sch, t_run=t_run)
#         return sch, t_run
#
#     return timed_scheduler


def runtime_wrapper(scheduler):
    @wraps(scheduler)
    def new_scheduler(tasks, ch_avail, runtimes):
        t_start = perf_counter()
        sch = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        # sch, loss, t_run = eval_wrapper(scheduler)(tasks, ch_avail)
        for runtime in runtimes:
            if t_run < runtime:
                yield sch
            else:
                yield None
                # raise RuntimeError(f"Algorithm timeout: {t_run} > {runtime}.")

    return new_scheduler


def evaluate_algorithms_runtime(
    algorithms,
    runtimes,
    problem_gen,
    n_gen=1,
    solve=False,
    verbose=0,
    plotting=0,
    save_path=None,
):
    loss_iter = np.array(
        [[tuple([np.nan] * alg["n_iter"] for alg in algorithms)] * n_gen]
        * len(runtimes),
        dtype=[(alg["name"], float, (alg["n_iter"],)) for alg in algorithms],
    )
    loss_mean = np.array(
        [[(np.nan,) * len(algorithms)] * n_gen] * len(runtimes),
        dtype=[(alg["name"], float) for alg in algorithms],
    )

    loss_opt = np.full(n_gen, np.nan)
    t_run_opt = np.full(n_gen, np.nan)  # TODO: use in plots?

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose, save_path)):
        if solve:
            (tasks, ch_avail), (sch, loss, t_run) = out_gen
            loss_opt[i_gen] = loss
            t_run_opt[i_gen] = t_run
        else:
            tasks, ch_avail = out_gen

        for name, func, n_iter in algorithms:
            for iter_ in range(n_iter):  # perform new algorithm runs
                if verbose >= 2:
                    print(f"  {name}: Iteration: {iter_ + 1}/{n_iter}", end="\r")

                # Evaluate schedule
                for i_time, sch in enumerate(func(tasks, ch_avail, runtimes)):
                    if sch is None:
                        continue  # TODO

                    check_schedule(tasks, sch)
                    loss = evaluate_schedule(tasks, sch)

                    loss_iter[name][i_time, i_gen, iter_] = loss

                    if plotting >= 3:
                        plot_schedule(tasks, sch, ch_avail, loss, name, ax=None)

            loss_mean[name][:, i_gen] = loss_iter[name][:, i_gen].mean(-1)

        if plotting >= 2:
            _, ax_gen = plt.subplots(
                2, 1, num=f"Scheduling Problem: {i_gen + 1}", clear=True
            )
            plot_task_losses(tasks, ax=ax_gen[0])
            plot_loss_runtime(runtimes, loss_iter[:, i_gen], do_std=False, ax=ax_gen[1])

    # Results
    if plotting >= 1:
        _, ax_results = plt.subplots(num="Results", clear=True)
        plot_loss_runtime(
            runtimes,
            loss_mean,
            do_std=True,
            ax=ax_results,
            # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
        )

    if solve:  # relative to B&B
        loss_mean_rel = loss_mean.copy()
        for name in algorithms["name"]:
            loss_mean_rel[name] -= loss_opt
            # loss_mean_rel[name] /= loss_opt

        if plotting >= 1:
            _, ax_results_rel = plt.subplots(num="Results (Relative)", clear=True)
            plot_loss_runtime(
                runtimes,
                loss_mean_rel,
                do_std=True,
                ax=ax_results_rel,
                ax_kwargs={
                    "ylabel": "Excess Loss",
                    # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                },
            )

    return loss_iter, loss_opt


# # Evaluation example
# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     ('Random', runtime_wrapper(algs.free.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs.free.earliest_release), 1),
#     ('MCTS', partial(algs.limit.mcts, verbose=False), 5),
#     ('Policy', runtime_wrapper(policy_model), 5),
#     # ('DQN Agent', dqn_agent, 5),
# ], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])
#
# runtimes = np.logspace(-2, -1, 20, endpoint=False)
# evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=40, solve=True, verbose=2, plotting=1,
#                             save=False, file=None)

from functools import wraps
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

from task_scheduling.algorithms.util import timing_wrapper
from task_scheduling.tree_search import ScheduleNodeBound
from task_scheduling.util.info import plot_task_losses
from task_scheduling.util.results import check_schedule, evaluate_schedule, plot_schedule


def branch_bound(tasks: list, ch_avail: list, runtimes: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of tasks.Base
    ch_avail : list of float
        Channel availability times.
    runtimes : list of float
        Allotted algorithm runtime.
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
            if node_new.l_lo < node_best.l_ex:  # new node is not dominated
                if node_new.l_up < node_best.l_ex:
                    node_best = node_new.roll_out(inplace=False)  # roll-out a new best node
                    stack = [s for s in stack if s.l_lo < node_best.l_ex]  # cut dominated nodes

                stack.append(node_new)  # add new node to stack, LIFO

            # Check run conditions
            if perf_counter() - t_run >= runtimes[i_time]:
                yield node_best.t_ex, node_best.ch_ex
                i_time += 1
                if i_time == n_times:
                    break

        if verbose:
            # progress = 1 - sum(math.factorial(len(node.seq_rem)) for node in stack) / math.factorial(len(tasks))
            # print(f'Search progress: {100*progress:.1f}% - Loss < {node_best.l_ex:.3f}', end='\r')
            print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

    for _ in range(i_time, n_times):
        yield node_best.t_ex, node_best.ch_ex

#%% MCTS WIP

# def mcts(tasks, ch_avail, runtimes, c_explore=0., visit_threshold=0, verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : Sequence of task_scheduling.tasks.Base
#     ch_avail : Sequence of float
#         Channel availability times.
#     runtimes : float or Sequence of float
#             Allotted algorithm runtimes.
#     c_explore : float, optional
#         Exploration weight. Higher values prioritize less frequently visited notes.
#     visit_threshold : int, optional
#         Nodes with up to this number of visits will select children using the `expansion` method.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng : int or RandomState or Generator, optional
#         NumPy random number generator or seed. Instance RNG if None.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
#
#     # node = ScheduleNode(tasks, ch_avail, rng=rng)
#     # node = node.mcts(runtimes, c_explore, visit_threshold, inplace=False, verbose=verbose)
#     # return node.t_ex, node.ch_ex
#     raise NotImplementedError


# def mcts_v1(tasks, ch_avail, runtimes, c_explore=1., verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : Sequence of task_scheduling.tasks.Base
#     ch_avail : Sequence of float
#         Channel availability times.
#     runtimes : float or Sequence of float
#             Allotted algorithm runtimes.
#     c_explore : float, optional
#         Exploration weight. Higher values prioritize unexplored tree nodes.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng : int or RandomState or Generator, optional
#         NumPy random number generator or seed. Instance RNG if None.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
#
#     # node = ScheduleNode(tasks, ch_avail, rng=rng)
#     # node = node.mcts_v1(runtimes, c_explore, inplace=False, verbose=verbose)
#     # return node.t_ex, node.ch_ex
#     raise NotImplementedError


# def mcts(tasks: list, ch_avail: list, runtimes: list, verbose=False):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : list of tasks.Base
#     ch_avail : list of float
#         Channel availability times.
#     runtimes : list of float
#         Allotted algorithm runtime.
#     verbose : bool
#         Enables printing of algorithm state information.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
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
#         loss = node.l_ex
#         tree.backup(seq, loss)  # update search tree from leaf sequence to root
#
#         if loss < loss_min:
#             node_best, loss_min = node, loss
#
#         if perf_counter() - t_run >= runtimes[i_time]:
#             yield node_best.t_ex, node_best.ch_ex
#             i_time += 1

# def mcts_orig(tasks: list, ch_avail: list, max_runtime=float('inf'), n_mc=None, verbose=False, rng=None):
#     """
#     Monte Carlo tree search algorithm.
#
#     Parameters
#     ----------
#     tasks : list of task_scheduling.tasks.Base
#     ch_avail : list of float
#         Channel availability times.
#     n_mc : int or list of int
#         Number of Monte Carlo roll-outs per task.
#     max_runtime : float
#         Allotted algorithm runtime.
#     verbose : bool
#         Enables printing of algorithm state information.
#     rng
#         NumPy random number generator or seed. Default Generator if None.
#
#     Returns
#     -------
#     t_ex : ndarray
#         Task execution times.
#     ch_ex : ndarray
#         Task execution channels.
#
#     """
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
#             if node_mc.l_ex < node_best.l_ex:  # Update best node
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
#     return node_best.t_ex, node_best.ch_ex


def plot_loss_runtime(t_run, l_ex, do_std=False, ax=None, ax_kwargs=None):
    """
    Line plot of total execution loss versus maximum runtime.

    Parameters
    ----------
    t_run : ndarray
        Runtime of algorithm.
    l_ex : ndarray
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

    names = l_ex.dtype.names
    for i_name, name in enumerate(names):
        l_mean = l_ex[name].mean(-1)
        ax.plot(t_run, l_mean, label=name)
        if do_std:
            l_std = l_ex[name].std(-1)
            # ax.errorbar(t_run, l_mean, yerr=l_std, label=name, errorevery=(i_name, len(names)))
            ax.fill_between(t_run, l_mean - l_std, l_mean + l_std, alpha=0.25)
        # else:
        #     ax.plot(t_run, l_mean, label=name)

    ax.set(xlabel='Runtime (s)', ylabel='Loss')
    ax.legend()
    ax.set(**ax_kwargs)


def runtime_wrapper(scheduler):
    @wraps(scheduler)
    def new_scheduler(tasks, ch_avail, runtimes):
        t_ex, ch_ex, t_run = timing_wrapper(scheduler)(tasks, ch_avail)
        for runtime in runtimes:
            if t_run < runtime:
                yield t_ex, ch_ex
            else:
                yield None
                # raise RuntimeError(f"Algorithm timeout: {t_run} > {runtime}.")

    return new_scheduler


def evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0,
                                save_path=None):

    l_ex_iter = np.array([[tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen] * len(runtimes),
                         dtype=[(alg['name'], float, (alg['n_iter'],)) for alg in algorithms])
    l_ex_mean = np.array([[(np.nan,) * len(algorithms)] * n_gen] * len(runtimes),
                         dtype=[(alg['name'], float) for alg in algorithms])

    l_ex_opt = np.full(n_gen, np.nan)
    t_run_opt = np.full(n_gen, np.nan)  # TODO: use in plots?

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose, save_path)):
        if solve:
            (tasks, ch_avail), (t_ex, ch_ex, t_run) = out_gen
            check_schedule(tasks, t_ex, ch_ex)
            l_ex_opt[i_gen] = evaluate_schedule(tasks, t_ex)
            t_run_opt[i_gen] = t_run
        else:
            tasks, ch_avail = out_gen

        for name, func, n_iter in algorithms:
            for iter_ in range(n_iter):  # perform new algorithm runs
                if verbose >= 2:
                    print(f'  {name}: Iteration: {iter_ + 1}/{n_iter}', end='\r')

                # Evaluate schedule
                for i_time, solution in enumerate(func(tasks, ch_avail, runtimes)):
                    if solution is None:
                        continue  # TODO

                    t_ex, ch_ex = solution

                    check_schedule(tasks, t_ex, ch_ex)
                    l_ex = evaluate_schedule(tasks, t_ex)

                    l_ex_iter[name][i_time, i_gen, iter_] = l_ex

                    if plotting >= 3:
                        plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

            l_ex_mean[name][:, i_gen] = l_ex_iter[name][:, i_gen].mean(-1)

        if plotting >= 2:
            _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
            plot_task_losses(tasks, ax=ax_gen[0])
            plot_loss_runtime(runtimes, l_ex_iter[:, i_gen], do_std=False, ax=ax_gen[1])

    # Results
    if plotting >= 1:
        _, ax_results = plt.subplots(num='Results', clear=True)
        plot_loss_runtime(runtimes, l_ex_mean, do_std=True,
                          ax=ax_results,
                          # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                          )

    if solve:  # relative to B&B
        l_ex_mean_rel = l_ex_mean.copy()
        for name in algorithms['name']:
            l_ex_mean_rel[name] -= l_ex_opt
            # l_ex_mean_rel[name] /= l_ex_opt

        if plotting >= 1:
            _, ax_results_rel = plt.subplots(num='Results (Relative)', clear=True)
            plot_loss_runtime(runtimes, l_ex_mean_rel, do_std=True,
                              ax=ax_results_rel,
                              ax_kwargs={'ylabel': 'Excess Loss',
                                         # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                         }
                              )

    return l_ex_iter, l_ex_opt


# #%% Evaluation example
# algorithms = np.array([
#     # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
#     ('Random', runtime_wrapper(algs.free.random_sequencer), 20),
#     ('ERT', runtime_wrapper(algs.free.earliest_release), 1),
#     ('MCTS', partial(algs.limit.mcts, verbose=False), 5),
#     ('Policy', runtime_wrapper(policy_model), 5),
#     # ('DQN Agent', dqn_agent, 5),
# ], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
#
# runtimes = np.logspace(-2, -1, 20, endpoint=False)
# evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=40, solve=True, verbose=2, plotting=1,
#                             save=False, file=None)

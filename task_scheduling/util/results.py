from time import perf_counter
from functools import wraps, partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_scheduling._core import SchedulingSolution, RandomGeneratorMixin as RNGMix
from task_scheduling.learning.base import Base as BaseLearningScheduler
# from task_scheduling.generators.scheduling_problems import Dataset


def check_schedule(tasks, t_ex, ch_ex, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : numpy.ndarray
        Task execution times.
    ch_ex : numpy.ndarray
        Task execution channels.
    tol : float, optional
        Time tolerance for validity conditions.

    Raises
    -------
    ValueError
        If tasks overlap in time.

    """

    # if np.isnan(t_ex).any():
    #     raise ValueError("All tasks must be scheduled.")

    for ch in np.unique(ch_ex):
        tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        t_ex_ch = t_ex[ch_ex == ch]
        for n_1 in range(len(tasks_ch)):
            if t_ex_ch[n_1] + tol < tasks_ch[n_1].t_release:
                raise ValueError("Tasks cannot be executed before their release time.")

            for n_2 in range(n_1 + 1, len(tasks_ch)):
                conditions = [t_ex_ch[n_1] + tol < t_ex_ch[n_2] + tasks_ch[n_2].duration,
                              t_ex_ch[n_2] + tol < t_ex_ch[n_1] + tasks_ch[n_1].duration]
                if all(conditions):
                    raise ValueError('Invalid Solution: Scheduling Conflict')


def evaluate_schedule(tasks, t_ex):
    """
    Evaluate scheduling loss.

    Parameters
    ----------
    tasks : Sequence of task_scheduling.tasks.Base
        Tasks
    t_ex : Sequence of float
        Task execution times.

    Returns
    -------
    float
        Total loss of scheduled tasks.

    """

    l_ex = 0.
    for task, t_ex in zip(tasks, t_ex):
        l_ex += task(t_ex)

    return l_ex


def eval_wrapper(scheduler):
    """Wraps a scheduler, creates a function that outputs runtime in addition to schedule."""

    @wraps(scheduler)
    def timed_scheduler(tasks, ch_avail):
        t_start = perf_counter()
        t_ex, ch_ex, *__ = scheduler(tasks, ch_avail)
        t_run = perf_counter() - t_start

        check_schedule(tasks, t_ex, ch_ex)
        l_ex = evaluate_schedule(tasks, t_ex)

        return SchedulingSolution(t_ex, ch_ex, l_ex, t_run)

    return timed_scheduler


def plot_schedule(tasks, t_ex, ch_ex, l_ex=None, name=None, ax=None, ax_kwargs=None):
    """
    Plot task schedule.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : numpy.ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : numpy.ndarray
        Task execution channels. NaN for unscheduled.
    l_ex : float or None
        Total loss of scheduled tasks.
    name : str or None
        Algorithm string representation
    ax : Axes or None
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """
    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    n_ch = len(np.unique(ch_ex))
    bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ax.broken_barh([(t_ex[n], tasks[n].duration) for n in range(len(tasks))], (-0.5, 1), facecolors=bar_colors)
    for n, task in enumerate(tasks):
        label = str(task)
        # label = f'Task #{n}'
        ax.broken_barh([(t_ex[n], task.duration)], (ch_ex[n] - 0.5, 1),
                       facecolors=bar_colors[n % len(bar_colors)], edgecolor='black', label=label)

    x_lim = min(t_ex), max(t_ex[n] + task.duration for n, task in enumerate(tasks))
    ax.set(xlim=x_lim, ylim=(-.5, n_ch - 1 + .5), xlabel='t',
           yticks=list(range(n_ch)), ylabel='Channel')

    ax.legend()

    _temp = []
    if isinstance(name, str):
        _temp.append(name)
    if l_ex is not None:
        _temp.append(f'Loss = {l_ex:.3f}')
    title = ', '.join(_temp)
    if len(title) > 0:
        ax.set_title(title)

    ax.set(**ax_kwargs)


def scatter_loss_runtime(t_run, l_ex, ax=None, ax_kwargs=None):
    """
    Scatter plot of total execution loss versus runtime.

    Parameters
    ----------
    t_run : numpy.ndarray
        Runtime of algorithm.
    l_ex : numpy.ndarray
        Total loss of scheduled tasks.
    ax : Axes or None
        Matplotlib axes target object.
    ax_kwargs : dict
        Additional Axes keyword parameters.

    """
    if ax is None:
        _, ax = plt.subplots()

    if ax_kwargs is None:
        ax_kwargs = {}

    for name in t_run.dtype.names:
        kwargs = {}
        if name == 'BB Optimal':
            kwargs.update(c='k')

        ax.scatter(1e3 * t_run[name], l_ex[name], label=name, **kwargs)

    ax.set(xlabel='Runtime (ms)', ylabel='Loss')
    ax.legend()
    ax.set(**ax_kwargs)


#%% Utilities
def _iter_to_mean(array):
    return np.array([tuple(map(np.mean, item)) for item in array.flatten()],
                    dtype=[(name, float) for name in array.dtype.names]).reshape(array.shape)


def _struct_mean(array):
    array = _iter_to_mean(array)
    data = tuple(array[name].mean() for name in array.dtype.names)
    return np.array(data, dtype=array.dtype)


def _add_opt(algorithms):
    if 'BB Optimal' not in algorithms['name']:
        _opt = np.array([('BB Optimal', None, 1)], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])
        algorithms = np.concatenate((_opt, algorithms))

    return algorithms


def _empty_result(algorithms, n):
    return np.array([(np.nan,) * len(algorithms)] * n, dtype=[(alg['name'], float) for alg in algorithms])


def _relative_loss(l_ex):
    names = l_ex.dtype.names
    if 'BB Optimal' not in names:
        raise ValueError("Optimal solutions must be included in the loss array.")

    l_ex_rel = l_ex.copy()
    for name in names:
        l_ex_rel[name] -= l_ex['BB Optimal']
        # l_ex_rel[name] /= l_ex_mean_opt

    return l_ex_rel


def _scatter_results(t_run, l_ex, label='Results', do_relative=False):

    __, ax_results = plt.subplots(num=label, clear=True)
    scatter_loss_runtime(t_run, l_ex,
                         ax=ax_results,
                         # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                         )

    if do_relative:  # relative to B&B
        l_ex_rel = _relative_loss(l_ex)

        # __, ax_results_rel = plt.subplots(num=f'{label} (Relative)', clear=True)
        # scatter_loss_runtime(t_run, l_ex_rel,
        #                      ax=ax_results_rel,
        #                      ax_kwargs={'ylabel': 'Excess Loss',
        #                                 # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
        #                                 }
        #                      )

        names = list(l_ex.dtype.names)
        names.remove('BB Optimal')
        __, ax_results_rel = plt.subplots(num=f'{label} (Relative)', clear=True)
        scatter_loss_runtime(t_run[names], l_ex_rel[names],
                             ax=ax_results_rel,
                             ax_kwargs={'ylabel': 'Excess Loss',
                                        # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                        }
                             )


def _print_averages(l_ex, t_run, log_path=None, do_relative=False):
    names = list(l_ex.dtype.names)

    # data = [[l_ex[name].mean(), t_run[name].mean()] for name in names]
    # columns = ['Loss', 'Runtime (s)']
    data = [[l_ex[name].mean(), 1e3 * t_run[name].mean()] for name in names]
    columns = ['Loss', 'Runtime (ms)']

    if do_relative:
        l_ex_rel = _relative_loss(l_ex)
        # for item, name in zip(data, names):
        #     item.insert(0, l_ex_rel[name].mean())
        # columns.insert(0, 'Excess Loss')
        l_ex_opt = data[names.index('BB Optimal')][0]
        for item, name in zip(data, names):
            item.insert(0, l_ex_rel[name].mean() / l_ex_opt)
        columns.insert(0, 'Excess Loss (%)')

    df = pd.DataFrame(data, index=pd.CategoricalIndex(names), columns=columns)
    df_str = df.to_markdown(tablefmt='github', floatfmt='.3f')

    print(df_str, end='\n\n')
    if log_path is not None:
        with open(log_path, 'a') as fid:
            print(df_str, end='\n\n', file=fid)


def _seed_to_rng(algorithms):
    """Convert algorithm `rng` arguments to NumPy `Generator` objects. Repeated calls to algorithms will use the RNG
    in-place, avoiding exact reproduction and ensuring new output for Monte Carlo evaluation."""
    for algorithm in algorithms:
        func = algorithm['func']
        if isinstance(func, partial) and 'rng' in func.keywords:
            func.keywords['rng'] = RNGMix.make_rng(func.keywords['rng'])


#%% Algorithm evaluation
def evaluate_algorithms_single(algorithms, problem, solution_opt=None, verbose=0, plotting=0, log_path=None):

    _seed_to_rng(algorithms)

    solve = solution_opt is not None
    if solve:
        algorithms = _add_opt(algorithms)

    _array_iter = np.array(tuple([np.nan] * alg['n_iter'] for alg in algorithms),
                           dtype=[(alg['name'], float, (alg['n_iter'],)) for alg in algorithms])
    l_ex_iter, t_run_iter = _array_iter.copy(), _array_iter.copy()

    for i_alg, (name, func, n_iter) in enumerate(algorithms):
        if verbose >= 1:
            print(f'{name} ({i_alg + 1}/{len(algorithms)})', end=('\r' if verbose == 1 else '\n'))

        for iter_ in range(n_iter):  # perform new algorithm runs
            if verbose >= 2:
                print(f'Iteration: {iter_ + 1}/{n_iter})', end='\r')

            # Run algorithm
            if name == 'BB Optimal':
                solution = solution_opt
            else:
                solution = eval_wrapper(func)(problem.tasks, problem.ch_avail)

            l_ex_iter[name][iter_] = solution.l_ex
            t_run_iter[name][iter_] = solution.t_run

            if plotting >= 2:
                plot_schedule(problem.tasks, solution.t_ex, solution.ch_ex, l_ex=solution.l_ex, name=name, ax=None)

            # if name == 'BB Optimal':
            #     solution = solution_opt
            # else:
            #     solution = timing_wrapper(func)(tasks, ch_avail)
            #
            # # t_ex, ch_ex, t_run = solution
            # t_ex, ch_ex, _l_ex, t_run = solution
            #
            # # Evaluate schedule
            # check_schedule(tasks, t_ex, ch_ex)
            # l_ex = evaluate_schedule(tasks, t_ex)
            #
            # l_ex_iter[name][iter_] = l_ex
            # t_run_iter[name][iter_] = t_run
            #
            # if plotting >= 2:
            #     plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

    # Results
    if plotting >= 1:
        _scatter_results(t_run_iter, l_ex_iter, label='Problem', do_relative=solve)
    if verbose >= 1:
        _print_averages(l_ex_iter, l_ex_iter, log_path, do_relative=solve)

    return l_ex_iter, t_run_iter


def evaluate_algorithms_gen(algorithms, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, log_path=None):
    """
    Compare scheduling algorithms for numerous sets of tasks and channel availabilities.

    Parameters
    ----------
    algorithms: iterable of callable
        Scheduling algorithms
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generator
    n_gen : int
        Number of scheduling problems to generate.
    solve : bool, optional
        Enables generation of Branch & Bound optimal solutions.
    verbose : int, optional
        Print level. '0' is silent, '1' prints iteration and average results, '2' prints solver and algorithm progress.
    plotting : int, optional
        Plotting level. '0' plots nothing, '1' plots average results, '2 plots every problem, '3' plots every iteration.
    log_path : PathLike, optional
        File path for logging of algorithm performance.

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    # if isinstance(problem_gen, Dataset) and n_gen > problem_gen.n_problems:  # avoid redundant computation
    #     n_gen = problem_gen.n_problems
    #     warn(f"Dataset cannot generate requested number of unique problems. Argument `n_gen` reduced to {n_gen}")

    _seed_to_rng(algorithms)

    if solve:
        algorithms = _add_opt(algorithms)

    l_ex_mean, t_run_mean = _empty_result(algorithms, n_gen), _empty_result(algorithms, n_gen)

    if verbose >= 1:
        print("Evaluating algorithms...")
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose)):
        if solve:
            problem, solution_opt = out_gen
        else:
            problem, solution_opt = out_gen, None

        l_ex_iter, t_run_iter = evaluate_algorithms_single(algorithms, problem, solution_opt, verbose - 1,
                                                           plotting - 1)
        l_ex_mean[i_gen], t_run_mean[i_gen] = map(_iter_to_mean, (l_ex_iter, t_run_iter))

    # Results
    if plotting >= 1:
        _scatter_results(t_run_mean, l_ex_mean, label='Gen', do_relative=solve)
    if verbose >= 1:
        if log_path is not None:
            with open(log_path, 'a') as fid:
                print(f'n_gen = {n_gen}', end='\n\n', file=fid)
        else:
            print(f'n_gen = {n_gen}', end='\n\n')

        _print_averages(l_ex_mean, t_run_mean, log_path, do_relative=solve)

    return l_ex_mean, t_run_mean


def evaluate_algorithms_train(algorithms, n_gen_learn, problem_gen, n_gen=1, n_mc=1, solve=False, verbose=0, plotting=0,
                              log_path=None):

    if sum(isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms) > 1:
        raise NotImplementedError("Currently supports only a single learner. "
                                  "See https://spork.nre.navy.mil/nrl-radar/CRM/task-scheduling/-/issues/8")

    # reuse_data = False
    # if isinstance(problem_gen, Dataset):
    #     n_gen_total = n_gen + n_gen_learn
    #     if problem_gen.repeat:
    #         reuse_data = True
    #         if n_gen_total > problem_gen.n_problems:
    #             raise ValueError("Dataset cannot generate enough unique problems.")
    #     else:
    #         if n_gen_total * n_mc > problem_gen.n_problems:
    #             raise ValueError("Dataset cannot generate enough problems.")

    _seed_to_rng(algorithms)

    if solve:
        algorithms = _add_opt(algorithms)

    l_ex_mc, t_run_mc = _empty_result(algorithms, n_mc), _empty_result(algorithms, n_mc)

    for i_mc in range(n_mc):
        if verbose >= 1:
            print(f"Train/test iteration: {i_mc + 1}/{n_mc}")

        # if reuse_data:
        #     problem_gen.shuffle()  # random train/test split

        if hasattr(problem_gen, 'repeat') and problem_gen.repeat:  # repeating `Dataset` problem generator
        # if isinstance(problem_gen, Dataset) and problem_gen.repeat:  # repeating `Dataset` problem generator
            problem_gen.shuffle()

        # Reset/train supervised learners
        for learner in algorithms['func']:
            if isinstance(learner, BaseLearningScheduler):
                learner.reset()
                learner.learn(n_gen_learn, verbose=verbose - 1)  # calls `problem_gen` via environment `reset`

        # Evaluate performance
        l_ex_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, solve,
                                                        verbose - 1, plotting - 1)
        l_ex_mc[i_mc], t_run_mc[i_mc] = _struct_mean(l_ex_mean), _struct_mean(t_run_mean)

    # Results
    if plotting >= 1:
        _scatter_results(t_run_mc, l_ex_mc, label='Train', do_relative=solve)
    if verbose >= 1:
        if log_path is not None:
            with open(log_path, 'a') as fid:
                print(f'- n_mc = {n_mc}\n- n_gen = {n_gen}', end='\n\n', file=fid)
        else:
            print(f'- n_mc = {n_mc}\n- n_gen = {n_gen}', end='\n\n')

        _print_averages(l_ex_mc, t_run_mc, log_path, do_relative=solve)

    return l_ex_mc, t_run_mc

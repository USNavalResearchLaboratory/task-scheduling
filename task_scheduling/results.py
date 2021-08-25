from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_scheduling.base import RandomGeneratorMixin as RNGMix
from task_scheduling.util import eval_wrapper, plot_schedule
from task_scheduling.learning.base import Base as BaseLearningScheduler
from task_scheduling.learning.supervised.base import Base as BaseSupervisedScheduler
from task_scheduling.generators.problems import Dataset


OPT_NAME = 'BB Optimal'


def _scatter_loss_runtime(t_run, l_ex, ax=None, ax_kwargs=None):
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
        if name == OPT_NAME:
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
    if OPT_NAME not in algorithms['name']:
        _opt = np.array([(OPT_NAME, None, 1)], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])
        algorithms = np.concatenate((_opt, algorithms))

    return algorithms


def _empty_result(algorithms, n):
    return np.array([(np.nan,) * len(algorithms)] * n, dtype=[(alg['name'], float) for alg in algorithms])


def _relative_loss(l_ex):
    names = l_ex.dtype.names
    if OPT_NAME not in names:
        raise ValueError("Optimal solutions must be included in the loss array.")

    l_ex_rel = l_ex.copy()
    for name in names:
        l_ex_rel[name] -= l_ex[OPT_NAME]
        # l_ex_rel[name] /= l_ex_mean_opt

    return l_ex_rel


def _scatter_results(t_run, l_ex, label='Results', do_relative=False):

    __, ax_results = plt.subplots(num=label, clear=True)
    _scatter_loss_runtime(t_run, l_ex,
                          ax=ax_results,
                          # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                          )

    if do_relative:  # relative to B&B
        l_ex_rel = _relative_loss(l_ex)

        # __, ax_results_rel = plt.subplots(num=f'{label} (Relative)', clear=True)
        # _scatter_loss_runtime(t_run, l_ex_rel,
        #                      ax=ax_results_rel,
        #                      ax_kwargs={'ylabel': 'Excess Loss',
        #                                 # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
        #                                 }
        #                      )

        names = list(l_ex.dtype.names)
        names.remove(OPT_NAME)
        __, ax_results_rel = plt.subplots(num=f'{label} (Relative)', clear=True)
        _scatter_loss_runtime(t_run[names], l_ex_rel[names],
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
        l_ex_opt = data[names.index(OPT_NAME)][0]
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
            if name == OPT_NAME:
                solution = solution_opt
            else:
                solution = eval_wrapper(func)(problem.tasks, problem.ch_avail)

            l_ex_iter[name][iter_] = solution.l_ex
            t_run_iter[name][iter_] = solution.t_run

            if plotting >= 2:
                plot_schedule(problem.tasks, solution.t_ex, solution.ch_ex, l_ex=solution.l_ex, name=name, ax=None)

            # if name == OPT_NAME:
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
    problem_gen : generators.problems.Base
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

    if isinstance(problem_gen, Dataset) and n_gen > problem_gen.n_problems:  # avoid redundant computation
        # n_gen = problem_gen.n_problems
        # warn(f"Dataset cannot generate requested number of unique problems. Argument `n_gen` reduced to {n_gen}")
        raise ValueError(f"Dataset cannot generate requested number of unique problems.")

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

    # if sum(isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms) > 1:  # TODO
    #     raise NotImplementedError("Currently supports only a single learner. "
    #                               "See https://spork.nre.navy.mil/nrl-radar/CRM/task-scheduling/-/issues/8")

    reuse_data = False
    if isinstance(problem_gen, Dataset):
        n_gen_total = n_gen + n_gen_learn
        if problem_gen.repeat:
            reuse_data = True
            if n_gen_total > problem_gen.n_problems:
                raise ValueError("Dataset cannot generate enough unique problems.")
        else:
            if n_gen_total * n_mc > problem_gen.n_problems:
                raise ValueError("Dataset cannot generate enough problems.")

    _seed_to_rng(algorithms)

    if solve:
        algorithms = _add_opt(algorithms)

    learners = algorithms[[isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms]]
    supervised_learners = learners[[isinstance(alg['func'], BaseSupervisedScheduler) for alg in learners]]
    _do_sl = bool(len(supervised_learners))

    l_ex_mc, t_run_mc = _empty_result(algorithms, n_mc), _empty_result(algorithms, n_mc)
    for i_mc in range(n_mc):
        if verbose >= 1:
            print(f"Train/test iteration: {i_mc + 1}/{n_mc}")

        if reuse_data:
            problem_gen.shuffle()  # random train/test split

        # if hasattr(problem_gen, 'repeat') and problem_gen.repeat:  # repeating `Dataset` problem generator
        # if isinstance(problem_gen, Dataset) and problem_gen.repeat:  # repeating `Dataset` problem generator
        #     problem_gen.shuffle()

        # Get training problems, make solutions if needed for SL
        out_gen = list(problem_gen(n_gen_learn, solve=_do_sl))
        if _do_sl:
            problems, solutions = zip(*out_gen)
        else:
            problems, solutions = out_gen, None

        # Reset/train supervised learners
        for learner in learners:
            if verbose >= 2:
                print(f"Training learner: {learner['name']}")

            func = learner['func']
            func.reset()
            func.env.problem_gen = Dataset(problems, solutions)
            func.learn(n_gen_learn, verbose=verbose - 1)  # calls `problem_gen` via environment `reset`

        # # Reset/train supervised learners
        # for learner in algorithms['func']:
        #     if isinstance(learner, BaseLearningScheduler):
        #         learner.reset()
        #         learner.learn(n_gen_learn, verbose=verbose - 1)  # calls `problem_gen` via environment `reset`

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

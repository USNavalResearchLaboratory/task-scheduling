import logging
import pickle
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_scheduling.base import RandomGeneratorMixin as RNGMix
from task_scheduling.generators.problems import Dataset, Base as BaseProblemGenerator
from task_scheduling.mdp.base import BaseLearning as BaseLearningScheduler
from task_scheduling.mdp.supervised.base import Base as BaseSupervisedScheduler
from task_scheduling.util import eval_wrapper, plot_schedule

opt_name = 'BB Optimal'
pickle_figs = True

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
out_handler = logging.StreamHandler(stream=sys.stdout)
out_formatter = logging.Formatter('\n# %(asctime)s\n%(message)s\n', datefmt='%Y-%m-%d %H:%M:%S')
out_handler.setFormatter(out_formatter)
logger.addHandler(out_handler)


@contextmanager
def _file_logger(file, file_format):
    if file is not None:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file)
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        yield logger
        logger.removeHandler(file_handler)
    else:
        yield logger


def _log_and_fig(message, log_path, ax, img_path):
    file_format = '\n# %(asctime)s\n%(message)s\n'
    if img_path is not None:
        img_path = Path(img_path)
        img_path.parent.mkdir(parents=True, exist_ok=True)

        file_format += f"\n![]({img_path.absolute().as_posix()})\n"

        fig = ax.figure
        fig.savefig(img_path)
        if pickle_figs:
            mpl_file = img_path.parent / f"{img_path.stem}.mpl"
            with open(mpl_file, 'wb') as fid:
                pickle.dump(fig, fid)

    with _file_logger(log_path, file_format) as logger_:
        logger_.info(message)


def _log_helper(problem_obj, learners, loss, t_run, solve, log_path, ax, img_path, rng, n_gen_learn=None, n_mc=None):
    message = f'- Seed = {rng}'
    if n_gen_learn is not None:
        message += f'\n- Training problems: {n_gen_learn}'
    if n_mc is not None:
        message += f'\n- MC iterations: {n_mc}'

    if isinstance(problem_obj, BaseProblemGenerator):
        # message += f"\n\n## Problem:\n{problem_obj.summary()}"
        pass
    else:
        message += f"\n\n## Problem:\n{problem_obj}"

    if len(learners) > 0:
        message += "\n\n## Learners"
        for learner in learners:
            message += f"\n\n### {learner['name']}"
            message += f"\n{learner['func'].summary()}"

    message += '\n\n## Results'
    message += f"\n{_print_averages(loss, t_run, do_relative=solve)}"

    _log_and_fig(message, log_path, ax, img_path)


# Utilities
def _iter_to_mean(array):
    return np.array([tuple(map(np.mean, item)) for item in array.flatten()],
                    dtype=[(name, float) for name in array.dtype.names]).reshape(array.shape)


def _struct_mean(array):
    array = _iter_to_mean(array)
    data = tuple(array[name].mean() for name in array.dtype.names)
    return np.array(data, dtype=array.dtype)


def _add_opt(algorithms):
    if opt_name not in algorithms['name']:
        _opt = np.array([(opt_name, None, 1)], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])
        algorithms = np.concatenate((_opt, algorithms))

    return algorithms


def _empty_result(algorithms, n):
    return np.array([(np.nan,) * len(algorithms)] * n, dtype=[(alg['name'], float) for alg in algorithms])


# Printing/plotting
def _relative_loss(loss, normalize=False):
    names = loss.dtype.names
    if opt_name not in names:
        raise ValueError("Optimal solutions must be included in the loss array.")

    loss_rel = loss.copy()
    for name in names:
        loss_rel[name] -= loss[opt_name]
        if normalize:
            loss_rel[name] /= loss[opt_name]
            loss_rel[name] *= 100  # as percentage

    return loss_rel


def _scatter_loss_runtime(t_run, loss, ax=None, ax_kwargs=None):
    """
    Scatter plot of total execution loss versus runtime.

    Parameters
    ----------
    t_run : numpy.ndarray
        Runtime of algorithm.
    loss : numpy.ndarray
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
        if name == opt_name:
            kwargs.update(c='k')

        ax.scatter(1e3 * t_run[name], loss[name], label=name, **kwargs)

    ax.set(xlabel='Runtime (ms)', ylabel='Loss')
    ax.legend()
    ax.set(**ax_kwargs)


def _scatter_results(t_run, loss, label='Results', do_relative=False):
    __, ax_results = plt.subplots(num=label, clear=True)
    _scatter_loss_runtime(t_run, loss,
                          ax=ax_results,
                          # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                          )

    if do_relative:  # relative to B&B
        normalize = True
        # normalize = False

        loss_rel = _relative_loss(loss, normalize)

        names = list(loss.dtype.names)
        names.remove(opt_name)
        __, ax_results_rel = plt.subplots(num=f'{label} (Relative)', clear=True)
        _scatter_loss_runtime(t_run[names], loss_rel[names],
                              ax=ax_results_rel,
                              ax_kwargs={'ylabel': 'Excess Loss' + ' (%)' if normalize else '',
                                         # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                         }
                              )


def _print_averages(loss, t_run, do_relative=False):
    names = list(loss.dtype.names)

    data = [[loss[name].mean(), 1e3 * t_run[name].mean()] for name in names]
    columns = ['Loss', 'Runtime (ms)']

    if do_relative:
        loss_rel = _relative_loss(loss)

        loss_opt = data[names.index(opt_name)][0]
        for item, name in zip(data, names):
            item.insert(0, loss_rel[name].mean() / loss_opt * 100)  # normalize to percentage
        columns.insert(0, 'Excess Loss (%)')

    df = pd.DataFrame(data, index=pd.CategoricalIndex(names), columns=columns)
    df_str = df.to_markdown(tablefmt='github', floatfmt='.3f')

    return df_str

    # print(df_str, end='\n\n')
    # if log_path is not None:
    #     with open(log_path, 'a') as fid:
    #         print(df_str, end='\n\n', file=fid)


# RNG handling
def _set_algorithm_rng(algorithms, rng):
    """Makes algorithms into `functools.partial` objects, overwrites any existing `rng` arguments."""
    for algorithm in algorithms:
        if isinstance(algorithm['func'], partial):
            func_code = algorithm['func'].func.__code__
            arg_names = func_code.co_varnames[:func_code.co_argcount]
            if 'rng' in arg_names:  # algorithm takes `rng` argument, can be seeded
                algorithm['func'].keywords['rng'] = rng
        else:
            try:  # FIXME: hack. should be able to inspect `__call__` signature
                func_code = algorithm['func'].__code__
            except AttributeError:
                warn(f"RNG cannot be set for algorithm: {algorithm['name']}")
                continue
            arg_names = func_code.co_varnames[:func_code.co_argcount]
            if 'rng' in arg_names:  # algorithm takes `rng` argument, can be seeded
                algorithm['func'] = partial(algorithm['func'])
                algorithm['func'].keywords['rng'] = rng


def _seed_to_rng(algorithms):
    """Convert algorithm `rng` arguments to NumPy `Generator` objects. Repeated calls to algorithms will use the RNG
    in-place, avoiding exact reproduction and ensuring new output for Monte Carlo evaluation."""
    for func in algorithms['func']:
        if isinstance(func, partial) and 'rng' in func.keywords:
            func.keywords['rng'] = RNGMix.make_rng(func.keywords['rng'])


# Algorithm evaluation
def evaluate_algorithms_single(algorithms, problem, solution_opt=None, verbose=0, plotting=0, log_path=None,
                               img_path=None, rng=None):
    """
    Compare scheduling algorithms.

    Parameters
    ----------
    algorithms: iterable of callable
        Scheduling algorithms
    problem : SchedulingProblem
    solution_opt : SchedulingSolution, optional
        Optimal scheduling solution.
    verbose : int, optional
        Print level. '0' is silent; '1' prints current algorithm and logs results; '2' adds iteration count.
    plotting : int, optional
        Plotting level. '0' plots nothing; '1' plots loss-runtime results; '2' adds every schedule.
    log_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    img_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    learners = algorithms[[isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms]]

    # RNG control
    if rng is not None:
        _set_algorithm_rng(algorithms, rng)
    _seed_to_rng(algorithms)

    solve = solution_opt is not None
    if solve:
        algorithms = _add_opt(algorithms)

    _array_iter = np.array(tuple([np.nan] * alg['n_iter'] for alg in algorithms),
                           dtype=[(alg['name'], float, (alg['n_iter'],)) for alg in algorithms])
    loss_iter, t_run_iter = _array_iter.copy(), _array_iter.copy()

    for i_alg, (name, func, n_iter) in enumerate(algorithms):
        if verbose >= 1:
            print(f'{name} ({i_alg + 1}/{len(algorithms)})', end=('\r' if verbose == 1 else '\n'))

        for iter_ in range(n_iter):  # perform new algorithm runs
            if verbose >= 2:
                print(f'Iteration: {iter_ + 1}/{n_iter})', end='\r')

            # Run algorithm
            if name == opt_name:
                solution = solution_opt
            else:
                solution = eval_wrapper(func)(problem.tasks, problem.ch_avail)

            loss_iter[name][iter_] = solution.loss
            t_run_iter[name][iter_] = solution.t_run

            if plotting >= 2:
                plot_schedule(problem.tasks, solution.sch, loss=solution.loss, name=name, ax=None)

    # Results
    if plotting >= 1:
        _scatter_results(t_run_iter, loss_iter, label='Problem', do_relative=solve)
        ax = plt.gca()
    else:
        ax, img_path = None, None

    # Logging
    if verbose >= 1:
        _log_helper(problem, learners, loss_iter, t_run_iter, solve, log_path, ax, img_path, rng)

    return loss_iter, t_run_iter


def evaluate_algorithms_gen(algorithms, problem_gen, n_gen=1, n_gen_learn=0, solve=False, verbose=0, plotting=0,
                            log_path=None, img_path=None, rng=None):
    """
    Compare scheduling algorithms against generated problems.

    Parameters
    ----------
    algorithms: iterable of callable
        Scheduling algorithms
    problem_gen : generators.problems.Base
        Scheduling problem generator
    n_gen : int, optional
        Number of scheduling problems to generate.
    n_gen_learn : int, optional
        Number of scheduling problems to generate training data from.
    solve : bool, optional
        Enables generation of Branch & Bound optimal solutions.
    verbose : int, optional
        Print level. '0' is silent; '1' prints learning status, problem count, and logs results;
        '2' adds individual problem info.
    plotting : int, optional
        Plotting level. '0' plots nothing; '1' plots average results; '2' adds every problem.
    log_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    img_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    learners = algorithms[[isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms]]
    _do_learn = bool(len(learners)) and bool(n_gen_learn)
    if not _do_learn:
        n_gen_learn = 0

    if isinstance(problem_gen, Dataset):
        if n_gen + n_gen_learn > problem_gen.n_problems:
            raise ValueError("Dataset cannot generate enough unique problems.")

    # RNG control
    if rng is not None:
        problem_gen.rng = rng
        if isinstance(problem_gen, Dataset):
            problem_gen.shuffle()

        _set_algorithm_rng(algorithms, rng)
    _seed_to_rng(algorithms)

    if solve:
        algorithms = _add_opt(algorithms)

    if _do_learn:
        # Get training problems, make solutions if needed for SL
        supervised_learners = learners[[isinstance(alg['func'], BaseSupervisedScheduler) for alg in learners]]
        _do_sl = bool(len(supervised_learners))

        out_gen = list(problem_gen(n_gen_learn, solve=_do_sl, verbose=verbose))
        if _do_sl:
            problems, solutions = zip(*out_gen)
        else:
            problems, solutions = out_gen, None

        # Reset/train supervised learners
        for learner in learners:
            if verbose >= 1:
                print(f"\nTraining learner: {learner['name']}")

            func = learner['func']
            func.reset()
            # instantiate new generator for each learner
            func.env.problem_gen = Dataset(problems, solutions, shuffle=True, repeat=True,
                                           task_gen=problem_gen.task_gen, ch_avail_gen=problem_gen.ch_avail_gen)
            func.learn(n_gen_learn, verbose=verbose)  # calls `problem_gen` via environment `reset`

    loss_mean, t_run_mean = _empty_result(algorithms, n_gen), _empty_result(algorithms, n_gen)
    if verbose >= 1:
        print("\nEvaluating algorithms...")
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose)):
        if solve:
            problem, solution_opt = out_gen
        else:
            problem, solution_opt = out_gen, None

        loss_iter, t_run_iter = evaluate_algorithms_single(algorithms, problem, solution_opt, verbose - 1, plotting - 1)
        loss_mean[i_gen], t_run_mean[i_gen] = map(_iter_to_mean, (loss_iter, t_run_iter))

    # Results
    if plotting >= 1:
        _scatter_results(t_run_mean, loss_mean, label='Gen', do_relative=solve)
        ax = plt.gca()
    else:
        ax, img_path = None, None

    # Logging
    if verbose >= 1:
        _log_helper(problem_gen, learners, loss_mean, t_run_mean, solve, log_path, ax, img_path, rng, n_gen_learn)

    return loss_mean, t_run_mean


def evaluate_algorithms_train(algorithms, problem_gen, n_gen=1, n_gen_learn=0, n_mc=1, solve=False, verbose=0,
                              plotting=0, log_path=None, img_path=None, rng=None):
    """
    Compare scheduling algorithms with iterative training of learners.

    Parameters
    ----------
    algorithms: iterable of callable
        Scheduling algorithms
    problem_gen : generators.problems.Base
        Scheduling problem generator
    n_gen : int, optional
        Number of scheduling problems to generate for evaluation, per Monte Carlo iteration
    n_gen_learn : int, optional
        Number of scheduling problems to generate training data from, per Monte Carlo iteration.
    n_mc : int, optional
        Number of Monte Carlo iterations for learner training.
    solve : bool, optional
        Enables generation of Branch & Bound optimal solutions.
    verbose : int, optional
        Print level. '0' is silent, '1' prints iteration and logs results, '2' adds info from each problem set.
    plotting : int, optional
        Plotting level. '0' plots nothing, '1' plots average results, '2' adds plots for each problem set.
    log_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    img_path : os.PathLike or str, optional
        File path for logging of algorithm performance.
    rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    learners = algorithms[[isinstance(alg['func'], BaseLearningScheduler) for alg in algorithms]]
    if len(learners) == 0:
        n_gen_learn = 0

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

    # RNG control
    if rng is not None:
        problem_gen.rng = rng
        if isinstance(problem_gen, Dataset):
            problem_gen.shuffle()

        _set_algorithm_rng(algorithms, rng)
    _seed_to_rng(algorithms)

    if solve:
        algorithms = _add_opt(algorithms)

    loss_mc, t_run_mc = _empty_result(algorithms, n_mc), _empty_result(algorithms, n_mc)
    if verbose >= 1:
        print("\nPerforming Monte Carlo assessment...")
    for i_mc in range(n_mc):
        if verbose >= 1:
            print(f"  Train/test iteration: {i_mc + 1}/{n_mc}")

        if reuse_data:
            problem_gen.shuffle()  # random train/test split

        # Evaluate performance
        loss_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve,
                                                        verbose=verbose - 1, plotting=plotting - 1)
        loss_mc[i_mc], t_run_mc[i_mc] = _struct_mean(loss_mean), _struct_mean(t_run_mean)

    # Results
    if plotting >= 1:
        _scatter_results(t_run_mc, loss_mc, label='Train', do_relative=solve)
        ax = plt.gca()
    else:
        ax, img_path = None, None

    # Logging
    if verbose >= 1:
        _log_helper(problem_gen, learners, loss_mc, t_run_mc, solve, log_path, ax, img_path, rng, n_gen_learn, n_mc)

    return loss_mc, t_run_mc

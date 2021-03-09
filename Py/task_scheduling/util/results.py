import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task_scheduling.util.generic import timing_wrapper, reset_weights
from task_scheduling.util.plot import plot_task_losses, plot_schedule, scatter_loss_runtime, plot_loss_runtime
from task_scheduling.generators.scheduling_problems import Dataset


# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


#%% Schedule evaluation
def check_schedule(tasks, t_ex, ch_ex, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Base
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
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
    tasks : Iterable of task_scheduling.tasks.Base
    t_ex : Iterable of float
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


#%% Utilities
def _iter_to_mean(array):
    return np.array([tuple(map(np.mean, item)) for item in array.flatten()],
                    dtype=[(name, float) for name in array.dtype.names]).reshape(array.shape)


def _struct_mean(array):
    array = _iter_to_mean(array)
    data = tuple(array[name].mean() for name in array.dtype.names)
    return np.array(data, dtype=array.dtype)


def _add_bb(algorithms):
    if 'BB Optimal' not in algorithms['name']:
        _opt = np.array([('BB Optimal', None, 1)], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
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


def scatter_results(t_run, l_ex, label='Results', do_relative=False):

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


def print_averages(l_ex, t_run, log_path=None, do_relative=False):
    names = l_ex.dtype.names

    data = [[l_ex[name].mean(), t_run[name].mean()] for name in names]
    columns = ['Loss', 'Runtime']

    if do_relative:
        l_ex_rel = _relative_loss(l_ex)
        for item, name in zip(data, names):
            item.insert(0, l_ex_rel[name].mean())
        columns.insert(0, 'Relative Loss')

    df = pd.DataFrame(data, index=pd.CategoricalIndex(names), columns=columns)
    df_str = df.to_markdown(tablefmt='github', floatfmt='.3f')

    print(df_str, end='\n\n')
    if log_path is not None:
        with open(log_path, 'a') as fid:
            print(df_str, end='\n\n', file=fid)


#%% Algorithm evaluation
def evaluate_algorithms_single(algorithms, tasks, ch_avail, solution_opt=None, verbose=0, plotting=0, log_path=None):

    solve = solution_opt is not None
    if solve:
        algorithms = _add_bb(algorithms)

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
                t_ex, ch_ex, t_run = solution_opt
            else:
                t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)

            # Evaluate schedule
            check_schedule(tasks, t_ex, ch_ex)
            l_ex = evaluate_schedule(tasks, t_ex)

            l_ex_iter[name][iter_] = l_ex
            t_run_iter[name][iter_] = t_run

            if plotting >= 2:
                plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

    # Results
    if plotting >= 1:
        scatter_results(t_run_iter, l_ex_iter, label='Problem', do_relative=solve)
    if verbose >= 1:
        print_averages(l_ex_iter, l_ex_iter, log_path, do_relative=solve)

    return l_ex_iter, t_run_iter


def evaluate_algorithms(algorithms, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, log_path=None):
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

    if solve:
        algorithms = _add_bb(algorithms)

    l_ex_mean, t_run_mean = _empty_result(algorithms, n_gen), _empty_result(algorithms, n_gen)

    if verbose >= 1:
        print("Evaluating algorithms...")
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose)):

        if solve:
            (tasks, ch_avail), solution_opt = out_gen
        else:
            tasks, ch_avail = out_gen
            solution_opt = None

        l_ex_iter, t_run_iter = evaluate_algorithms_single(algorithms, tasks, ch_avail, solution_opt, verbose - 1,
                                                           plotting - 1)
        l_ex_mean[i_gen], t_run_mean[i_gen] = map(_iter_to_mean, (l_ex_iter, t_run_iter))

    # Results
    if plotting >= 1:
        scatter_results(t_run_mean, l_ex_mean, label='Gen', do_relative=solve)
    if verbose >= 1:
        if log_path is not None:
            with open(log_path, 'a') as fid:
                print(f'n_gen = {n_gen}', end='\n\n', file=fid)
        else:
            print(f'n_gen = {n_gen}', end='\n\n')

        print_averages(l_ex_mean, t_run_mean, log_path, do_relative=solve)

    return l_ex_mean, t_run_mean


def evaluate_algorithms_train(algorithms, train_args, problem_gen, n_gen=1, n_mc=1, solve=False, verbose=0, plotting=0,
                              log_path=None):

    if isinstance(problem_gen, Dataset):
        n_gen_train = (train_args['n_batch_train'] + train_args['n_batch_val']) * train_args['batch_size']
        n_gen_total = n_gen + n_gen_train
        if problem_gen.repeat:
            if n_gen_total > problem_gen.n_problems:
                raise ValueError("Dataset cannot generate enough unique problems.")
        else:
            if n_gen_total * n_mc > problem_gen.n_problems:
                raise ValueError("Dataset cannot generate enough problems.")

    reuse_data = isinstance(problem_gen, Dataset) and problem_gen.repeat

    if solve:
        algorithms = _add_bb(algorithms)

    l_ex_mc, t_run_mc = _empty_result(algorithms, n_mc), _empty_result(algorithms, n_mc)

    for i_mc in range(n_mc):
        if verbose >= 1:
            print(f"MC iteration {i_mc + 1}/{n_mc}")

        if reuse_data:
            problem_gen.shuffle()  # random train/test split

        # Reset/train supervised learner
        try:
            learner = algorithms['func'][algorithms['name'].tolist().index('NN')]
            reset_weights(learner.model)
            learner.learn(verbose=verbose - 1, **train_args)  # note: calls `problem_gen` via environment reset
            # TODO: generalize for multiple learners, ensure same data is used for each training op
        except ValueError:
            pass

        # Evaluate performance
        l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen, solve, verbose - 1, plotting - 1)
        l_ex_mc[i_mc], t_run_mc[i_mc] = _struct_mean(l_ex_mean), _struct_mean(t_run_mean)

    # Results
    if plotting >= 1:
        scatter_results(t_run_mc, l_ex_mc, label='Train', do_relative=solve)
    if verbose >= 1:
        if log_path is not None:
            with open(log_path, 'a') as fid:
                print(f'- n_mc = {n_mc}\n- n_gen = {n_gen}', end='\n\n', file=fid)
        else:
            print(f'- n_mc = {n_mc}\n- n_gen = {n_gen}', end='\n\n')

        print_averages(l_ex_mc, t_run_mc, log_path, do_relative=solve)

    return l_ex_mc, t_run_mc


#%% Runtime limited operation
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

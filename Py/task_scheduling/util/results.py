import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task_scheduling.util.generic import timing_wrapper, reset_weights
from task_scheduling.util.plot import plot_task_losses, plot_schedule, scatter_loss_runtime, plot_loss_runtime
from task_scheduling.generators.scheduling_problems import Dataset


# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


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


def iter_to_mean(array):    # TODO: rework eval funcs, deprecate?
    return np.array([tuple(map(np.mean, item)) for item in array.flatten()],
                    dtype=[(name, float) for name in array.dtype.names]).reshape(array.shape)


def evaluate_dat(algorithms, problem_gen, n_gen=1, n_mc=1, train_args=None, solve=False, verbose=0, plotting=0):
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
        Print level. '0' is silent, '1' prints iteration, '2' prints solver and algorithm progress.
    plotting : int, optional
        Plotting level. '0' plots nothing, '1' plots every problem, '2' plots every iteration.

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    if solve:
        _opt = np.array([('BB Optimal', None, 1)], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
        algorithms = np.concatenate((_opt, algorithms))

    _array_iter = np.array([[tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen] * n_mc,
                           dtype=[(alg['name'], float, (alg['n_iter'],)) for alg in algorithms])

    l_ex_iter = _array_iter.copy()
    t_run_iter = _array_iter.copy()

    if train_args is None:
        train_args = {'n_batch_train': 0, 'n_batch_val': 0, 'batch_size': 0}

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

    # Generate scheduling problems
    for i_mc in range(n_mc):
        print(f"MC iteration {i_mc + 1}/{n_mc}")

        if reuse_data:
            problem_gen.shuffle()  # random train/test split

        # Reset/train supervised learner
        for alg in algorithms:      # FIXME: ensure same data is used for each training op
            if 'NN' in alg['name']:
                reset_weights(alg['func'].model)
                alg['func'].learn(**train_args)  # note: calls `problem_gen` via environment reset

        for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose)):
            if solve:
                (tasks, ch_avail), solution_opt = out_gen
            else:
                tasks, ch_avail = out_gen
                solution_opt = None

            for name, func, n_iter in algorithms:
                for iter_ in range(n_iter):  # perform new algorithm runs
                    if verbose >= 2:
                        print(f'  {name} ({iter_ + 1}/{n_iter})', end='\r')

                    # Run algorithm
                    if name == 'BB Optimal':
                        t_ex, ch_ex, t_run = solution_opt
                    else:
                        t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)

                    # Evaluate schedule
                    check_schedule(tasks, t_ex, ch_ex)
                    l_ex = evaluate_schedule(tasks, t_ex)

                    l_ex_iter[name][i_mc, i_gen, iter_] = l_ex
                    t_run_iter[name][i_mc, i_gen, iter_] = t_run

                    if plotting >= 2:
                        plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

            if plotting >= 1:
                _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
                plot_task_losses(tasks, ax=ax_gen[0])
                scatter_loss_runtime(t_run_iter[i_mc, i_gen], l_ex_iter[i_mc, i_gen], ax=ax_gen[1])

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

    # TODO: change default behavior?

    l_ex_iter, t_run_iter = evaluate_dat(algorithms, problem_gen, n_gen, 1, None, solve, verbose, plotting - 1)
    # l_ex_iter, t_run_iter = evaluate_dat(algorithms, problem_gen, n_gen, solve, verbose, plotting - 1)
    l_ex_mean, t_run_mean = map(iter_to_mean, (l_ex_iter, t_run_iter))

    # Results
    if plotting >= 1:
        __, ax_results = plt.subplots(num='Results', clear=True)
        scatter_loss_runtime(t_run_mean, l_ex_mean,
                             ax=ax_results,
                             # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                             )

    if solve:  # relative to B&B
        l_ex_mean_rel = l_ex_mean.copy()
        l_ex_mean_rel['BB Optimal'] = 0.
        for name in algorithms['name']:
            l_ex_mean_rel[name] -= l_ex_mean['BB Optimal']
            # l_ex_mean_rel[name] /= l_ex_mean_opt

        if plotting >= 1:
            __, ax_results_rel = plt.subplots(num='Results (Relative)', clear=True)
            scatter_loss_runtime(t_run_mean, l_ex_mean_rel,
                                 ax=ax_results_rel,
                                 ax_kwargs={'ylabel': 'Excess Loss',
                                            # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                            }
                                 )

            __, ax_results_rel_no_bb = plt.subplots(num='Results (Relative, opt excluded)', clear=True)
            scatter_loss_runtime(t_run_mean[algorithms['name']], l_ex_mean_rel[algorithms['name']],
                                 ax=ax_results_rel_no_bb,
                                 ax_kwargs={'ylabel': 'Excess Loss',
                                            # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                            }
                                 )

    if verbose >= 1:    # TODO: move calc to _train func?
        _data = [[l_ex_mean[name].mean(), t_run_mean[name].mean()] for name in algorithms['name']]
        df = pd.DataFrame(_data, index=pd.CategoricalIndex(algorithms['name']), columns=['Loss', 'Runtime'])
        df_str = '\n' + df.to_markdown(tablefmt='github', floatfmt='.3f')

        print(df_str)
        if log_path is not None:
            with open(log_path, 'a') as fid:
                print(df_str, file=fid)

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

    _array = np.array([[(np.nan,) * len(algorithms)] * n_gen] * n_mc,
                      dtype=[(alg['name'], float) for alg in algorithms])

    l_ex_mc = _array.copy()
    t_run_mc = _array.copy()
    # l_ex_mc_rel = _array.copy()

    reuse_data = isinstance(problem_gen, Dataset) and problem_gen.repeat
    for i_mc in range(n_mc):
        print(f"MC iteration {i_mc + 1}/{n_mc}")

        if reuse_data:
            problem_gen.shuffle()  # random train/test split

        # Reset/train supervised learner
        _idx = algorithms['name'].tolist().index('NN')
        reset_weights(algorithms['func'][_idx].model)
        algorithms['func'][_idx].learn(**train_args)  # note: calls `problem_gen` via environment reset

        # Evaluate performance

        # l_ex_iter, t_run_iter = evaluate_dat(algorithms, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0)
        # l_ex_mean, t_run_mean = map(iter_to_mean, (l_ex_iter, t_run_iter))

        l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen, n_gen, solve, verbose, plotting, log_path)

        l_ex_mc[i_mc] = l_ex_mean
        t_run_mc[i_mc] = t_run_mean

        l_ex_mean_opt = l_ex_mean['BB Optimal'].copy()
        l_ex_mean_rel = l_ex_mean.copy()
        for name in algorithms['name']:
            l_ex_mc[name][i_mc] = l_ex_mean[name].mean()
            t_run_mc[name][i_mc] = t_run_mean[name].mean()

            l_ex_mean_rel[name] -= l_ex_mean_opt
            l_ex_mc_rel[name][i_mc] = l_ex_mean_rel[name].mean()

        # Plot
        __, ax_results_rel = plt.subplots(num='Results MC (Relative)', clear=True)
        scatter_loss_runtime(t_run_mc, l_ex_mc_rel,
                             ax=ax_results_rel,
                             ax_kwargs={'ylabel': 'Excess Loss',
                                        # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                        }
                             )


#%% Runtime limited operation
def evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0,
                                save_path=None):
    # if solve:
    #     _opt = np.array([('B&B Optimal', None, 1)], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
    #     algorithms = np.concatenate((_opt, algorithms))

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

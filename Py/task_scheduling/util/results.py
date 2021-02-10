import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from task_scheduling.util.generic import timing_wrapper
from task_scheduling.util.plot import plot_task_losses, plot_schedule, scatter_loss_runtime, plot_loss_runtime

# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


def check_valid(tasks, t_ex, ch_ex, tol=1e-12):
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


def eval_loss(tasks, t_ex):
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


def iter_to_mean(array):
    return np.array([tuple(map(np.mean, item)) for item in array],
                    dtype=[(name, float) for name in array.dtype.names])


def evaluate_algorithms(algorithms, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, data_path=None,
                        log_path=None, rng=None):
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
        Progress print-out level. '0' is silent, '1' prints average results, '2' prints for every problem,
        '3' prints for every iteration.
    plotting : int, optional
        Plotting level. '0' plots nothing, '1' plots average results, '2' plots for every problem.
    data_path : PathLike, optional
        File path for saving data.
    log_path : PathLike, optional
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

    if solve:
        _opt = np.array([('BB Optimal', None, 1)], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
        algorithms = np.concatenate((_opt, algorithms))

    _args_iter = {'object': [tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen,
                  'dtype': [(alg['name'], float, (alg['n_iter'],)) for alg in algorithms]}
    _args_mean = {'object': [(np.nan,) * len(algorithms)] * n_gen,
                  'dtype': [(alg['name'], float) for alg in algorithms]}

    l_ex_iter = np.array(**_args_iter)
    l_ex_mean = np.array(**_args_mean)

    t_run_iter = np.array(**_args_iter)
    t_run_mean = np.array(**_args_mean)

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose, data_path, rng)):
        if solve:
            (tasks, ch_avail), solution_opt = out_gen
        else:
            tasks, ch_avail = out_gen
            solution_opt = None

        for name, func, n_iter in algorithms:
            if verbose >= 2:
                print(f'  {name}', end='\n')
            for iter_ in range(n_iter):      # perform new algorithm runs
                if verbose >= 3:
                    print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')

                # Run algorithm
                if name == 'BB Optimal':
                    t_ex, ch_ex, t_run = solution_opt
                else:
                    t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)

                # Evaluate schedule
                check_valid(tasks, t_ex, ch_ex)
                l_ex = eval_loss(tasks, t_ex)

                l_ex_iter[name][i_gen, iter_] = l_ex
                t_run_iter[name][i_gen, iter_] = t_run

                if plotting >= 3:
                    plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

            l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean()
            t_run_mean[name][i_gen] = t_run_iter[name][i_gen].mean()

            if verbose >= 2:
                print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.3f}"
                      f"\n    Avg. Runtime: {t_run_mean[name][i_gen]:.3f} (s)")

        if plotting >= 2:
            _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
            plot_task_losses(tasks, ax=ax_gen[0])
            scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

    # Results
    if plotting >= 1:
        __, ax_results = plt.subplots(num='Results', clear=True)
        scatter_loss_runtime(t_run_mean, l_ex_mean,
                             ax=ax_results,
                             # ax_kwargs={'title': f'Performance, {problem_gen.n_tasks} tasks'}
                             )

    if verbose >= 1:
        _data = [[l_ex_mean[name].mean(), t_run_mean[name].mean()] for name in algorithms['name']]
        df = pd.DataFrame(_data, index=pd.CategoricalIndex(algorithms['name']), columns=['Loss', 'Runtime'])

        if log_path is None:
            print('\n' + df.to_markdown(tablefmt='github', floatfmt='.3f'))
        else:
            with open(log_path, 'a') as fid:
                print('\n' + df.to_markdown(tablefmt='github', floatfmt='.3f'), file=fid)

    if solve:   # relative to B&B
        l_ex_mean_opt = l_ex_mean['BB Optimal'].copy()

        l_ex_mean_norm = l_ex_mean.copy()
        names = algorithms['name']
        for name in names:
            l_ex_mean_norm[name] -= l_ex_mean_opt
            l_ex_mean_norm[name] /= l_ex_mean_opt
        if plotting >= 1:
            __, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
            scatter_loss_runtime(t_run_mean, l_ex_mean_norm,
                                 ax=ax_results_norm,
                                 ax_kwargs={'ylabel': 'Excess Loss (Normalized)',
                                            # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                            }
                                 )

            __, ax_results_norm_no_bb = plt.subplots(num='Results (Normalized), no BB', clear=True)
            scatter_loss_runtime(t_run_mean[names[1:]], l_ex_mean_norm[names[1:]],
                                 ax=ax_results_norm_no_bb,
                                 ax_kwargs={'ylabel': 'Excess Loss (Normalized)',
                                            # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                            }
                                 )

    return l_ex_iter, t_run_iter


def evaluate_algorithms_runtime(algorithms, runtimes, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0,
                                save_path=None, rng=None):

    # if solve:
    #     _opt = np.array([('B&B Optimal', None, 1)], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])
    #     algorithms = np.concatenate((_opt, algorithms))

    l_ex_iter = np.array([[tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen] * len(runtimes),
                         dtype=[(alg['name'], float, (alg['n_iter'],)) for alg in algorithms])
    l_ex_mean = np.array([[(np.nan,) * len(algorithms)] * n_gen] * len(runtimes),
                         dtype=[(alg['name'], float) for alg in algorithms])

    l_ex_opt = np.full(n_gen, np.nan)
    t_run_opt = np.full(n_gen, np.nan)  # TODO: use in plots

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose, save_path, rng)):
        if solve:
            (tasks, ch_avail), (t_ex, ch_ex, t_run) = out_gen
            check_valid(tasks, t_ex, ch_ex)
            l_ex_opt[i_gen] = eval_loss(tasks, t_ex)
            t_run_opt[i_gen] = t_run
        else:
            tasks, ch_avail = out_gen

        for name, func, n_iter in algorithms:
            # if verbose >= 2:
            #     print(f'  {name}', end='\n')

            for iter_ in range(n_iter):      # perform new algorithm runs
                if verbose >= 2:
                    # print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')
                    print(f'  {name}: Iteration: {iter_ + 1}/{n_iter}', end='\r')

                # Evaluate schedule
                for i_time, solution in enumerate(func(tasks, ch_avail, runtimes)):
                    if solution is None:
                        continue    # TODO

                    t_ex, ch_ex = solution

                    check_valid(tasks, t_ex, ch_ex)
                    l_ex = eval_loss(tasks, t_ex)

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

    if solve:   # relative to B&B
        l_ex_mean_norm = l_ex_mean.copy()
        for name in algorithms['name']:
            l_ex_mean_norm[name] -= l_ex_opt
            l_ex_mean_norm[name] /= l_ex_opt

        if plotting >= 1:
            _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
            plot_loss_runtime(runtimes, l_ex_mean_norm, do_std=True,
                              ax=ax_results_norm,
                              ax_kwargs={'ylabel': 'Excess Loss (Normalized)',
                                         # 'title': f'Relative performance, {problem_gen.n_tasks} tasks',
                                         }
                              )

    return l_ex_iter, l_ex_opt

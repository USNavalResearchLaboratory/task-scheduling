import numpy as np
import matplotlib.pyplot as plt

from task_scheduling.util.generic import timing_wrapper
from task_scheduling.util.plot import plot_task_losses, scatter_loss_runtime, plot_schedule

# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


def check_valid(tasks, t_ex, ch_ex, tol=1e-12):
    """
    Check schedule validity.

    Parameters
    ----------
    tasks : list of task_scheduling.tasks.Generic
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
    tasks : task_scheduling.tasks.Generic
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


def compare_algorithms(algorithms, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, save=False, file=None):
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
        Progress print-out level. '0' is silent, '1' prints average results,
        '2' prints for every problem, '3' prints for every iteration.
    plotting : int, optional
        Plotting level. '0' plots nothing, '1' plots average results, '2' plots for every problem.
    save : bool, optional
        Enables serialization of generated problems/solutions.
    file : str, optional
        File location relative to data/schedules/

    Returns
    -------
    ndarray
        Algorithm scheduling execution losses.
    ndarray
        Algorithm scheduling runtimes.

    """

    if solve:
        _args_iter = {'object': [([np.nan],) + tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen,
                      'dtype': [('B&B Optimal', np.float, (1,))] + [(alg['name'], np.float, (alg['n_iter'],))
                                                                    for alg in algorithms]}
        _args_mean = {'object': [(np.nan,) * (1 + len(algorithms))] * n_gen,
                      'dtype': [('B&B Optimal', np.float)] + [(alg['name'], np.float) for alg in algorithms]}
    else:
        _args_iter = {'object': [tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_gen,
                      'dtype': [(alg['name'], np.float, (alg['n_iter'],)) for alg in algorithms]}
        _args_mean = {'object': [(np.nan,) * len(algorithms)] * n_gen,
                      'dtype': [(alg['name'], np.float) for alg in algorithms]}

    l_ex_iter = np.array(**_args_iter)
    t_run_iter = np.array(**_args_iter)

    l_ex_mean = np.array(**_args_mean)
    t_run_mean = np.array(**_args_mean)

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose >= 1, save, file)):
        if solve:
            (tasks, ch_avail), (t_ex_opt, ch_ex_opt, t_run_opt) = out_gen

            check_valid(tasks, t_ex_opt, ch_ex_opt)
            l_ex_opt = eval_loss(tasks, t_ex_opt)

            l_ex_iter['B&B Optimal'][i_gen, 0] = l_ex_opt
            t_run_iter['B&B Optimal'][i_gen, 0] = t_run_opt

            l_ex_mean['B&B Optimal'][i_gen] = l_ex_opt
            t_run_mean['B&B Optimal'][i_gen] = t_run_opt

            if verbose >= 2:
                print(f'  B&B Optimal', end='\n')
                print(f"    Avg. Runtime: {t_run_mean['B&B Optimal'][i_gen]:.2f} (s)")
                print(f"    Avg. Execution Loss: {l_ex_mean['B&B Optimal'][i_gen]:.2f}")

        else:
            tasks, ch_avail = out_gen

        for alg_repr, alg_func, n_iter in algorithms:
            if verbose >= 2:
                print(f'  {alg_repr}', end='\n')
            for iter_ in range(n_iter):      # Perform new algorithm runs
                if verbose >= 3:
                    print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')

                # Run algorithm
                t_ex, ch_ex, t_run = timing_wrapper(alg_func)(tasks, ch_avail)

                # Evaluate schedule
                check_valid(tasks, t_ex, ch_ex)
                l_ex = eval_loss(tasks, t_ex)

                # Store loss and runtime
                l_ex_iter[alg_repr][i_gen, iter_] = l_ex
                t_run_iter[alg_repr][i_gen, iter_] = t_run

                if plotting >= 3:
                    plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

            l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean()
            t_run_mean[alg_repr][i_gen] = t_run_iter[alg_repr][i_gen].mean()

            if verbose >= 2:
                print(f"    Avg. Runtime: {t_run_mean[alg_repr][i_gen]:.2f} (s)")
                print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][i_gen]:.2f}")

        if plotting >= 2:
            _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
            plot_task_losses(tasks, ax=ax_gen[0])
            scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

    # Results
    if plotting >= 1:
        _, ax_results = plt.subplots(num='Results', clear=True)
        scatter_loss_runtime(t_run_mean, l_ex_mean,
                             ax=ax_results,
                             ax_kwargs={'title': f'Performance on sets of {problem_gen.n_tasks} tasks'})

    if verbose >= 1:
        print('\nAvg. Performance\n' + 16*'-')
        print(f"{'Algorithm:':<35}{'Loss:':<8}{'Runtime (s):':<10}")
        if solve:
            print(f"{'B&B Optimal':<35}{l_ex_mean['B&B Optimal'].mean():<8.2f}"
                  f"{t_run_mean['B&B Optimal'].mean():<10.6f}")
        for rep in algorithms['name']:
            print(f"{rep:<35}{l_ex_mean[rep].mean():<8.2f}{t_run_mean[rep].mean():<10.6f}")

    if solve:   # Relative to B&B
        l_ex_mean_opt = l_ex_mean['B&B Optimal'].copy()
        # t_run_mean_opt = t_run_mean['B&B Optimal'].copy()

        l_ex_mean_norm = l_ex_mean.copy()
        t_run_mean_norm = t_run_mean.copy()

        # t_run_mean_norm['B&B Optimal'] = 0.
        l_ex_mean_norm['B&B Optimal'] = 0.
        for rep in algorithms['name']:
            l_ex_mean_norm[rep] -= l_ex_mean_opt
            l_ex_mean_norm[rep] /= l_ex_mean_opt
            # t_run_mean_norm[rep] -= t_run_mean_opt
            # t_run_mean_norm[rep] /= t_run_mean_opt

        if plotting >= 1:
            _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
            scatter_loss_runtime(t_run_mean_norm, l_ex_mean_norm,
                                 ax=ax_results_norm,
                                 ax_kwargs={'title': f'Relative Performance on sets '
                                                     f'of {problem_gen.n_tasks} tasks',
                                            'ylabel': 'Excess Loss (Normalized)',
                                            # 'xlabel': 'Runtime Difference (Normalized)'
                                            }
                                 )

    return l_ex_iter, t_run_iter

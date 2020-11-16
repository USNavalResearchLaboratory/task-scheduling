from itertools import product

import numpy as np
import matplotlib.pyplot as plt

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
            name = 'B&B Optimal'
            (tasks, ch_avail), (t_ex_opt, ch_ex_opt, t_run_opt) = out_gen

            check_valid(tasks, t_ex_opt, ch_ex_opt)
            l_ex_opt = eval_loss(tasks, t_ex_opt)

            l_ex_iter[name][i_gen, 0] = l_ex_opt
            t_run_iter[name][i_gen, 0] = t_run_opt

            l_ex_mean[name][i_gen] = l_ex_opt
            t_run_mean[name][i_gen] = t_run_opt

            if verbose >= 2:
                print(f'  {name}', end='\n')
                print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.2f}")
                print(f"    Avg. Runtime: {t_run_mean[name][i_gen]:.2f} (s)")

        else:
            tasks, ch_avail = out_gen

        for name, func, n_iter in algorithms:
            if verbose >= 2:
                print(f'  {name}', end='\n')
            for iter_ in range(n_iter):      # Perform new algorithm runs
                if verbose >= 3:
                    print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')

                # Run algorithm
                t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)

                # Evaluate schedule
                check_valid(tasks, t_ex, ch_ex)
                l_ex = eval_loss(tasks, t_ex)

                # Store loss and runtime
                l_ex_iter[name][i_gen, iter_] = l_ex
                t_run_iter[name][i_gen, iter_] = t_run

                if plotting >= 3:
                    plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

            l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean()
            t_run_mean[name][i_gen] = t_run_iter[name][i_gen].mean()

            if verbose >= 2:
                print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.2f}")
                print(f"    Avg. Runtime: {t_run_mean[name][i_gen]:.2f} (s)")

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
        names = algorithms['name'].tolist()
        if solve:
            names.insert(0, 'B&B Optimal')
        for name in names:
            print(f"{name:<35}{l_ex_mean[name].mean():<8.2f}{t_run_mean[name].mean():<10.6f}")

    if solve:   # Relative to B&B
        l_ex_mean_opt = l_ex_mean['B&B Optimal'].copy()
        # t_run_mean_opt = t_run_mean['B&B Optimal'].copy()

        l_ex_mean_norm = l_ex_mean.copy()
        t_run_mean_norm = t_run_mean.copy()

        l_ex_mean_norm['B&B Optimal'] = 0.
        # t_run_mean_norm['B&B Optimal'] = 0.
        for name in algorithms['name']:
            l_ex_mean_norm[name] -= l_ex_mean_opt
            l_ex_mean_norm[name] /= l_ex_mean_opt
            # t_run_mean_norm[name] -= t_run_mean_opt
            # t_run_mean_norm[name] /= t_run_mean_opt

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


def compare_algorithms_lim(algorithms, runtimes, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, save=False, file=None):
    n_runtimes = len(runtimes)

    if solve:
        _args_iter = {'object': [[([np.nan],) + tuple([np.nan] * alg['n_iter']
                                                      for alg in algorithms)] * n_runtimes] * n_gen,
                      'dtype': [('B&B Optimal', np.float, (1,))] + [(alg['name'], np.float, (alg['n_iter'],))
                                                                    for alg in algorithms],
                      }
        _args_mean = {'object': [[(np.nan,) * (1 + len(algorithms))] * n_runtimes] * n_gen,
                      'dtype': [('B&B Optimal', np.float)] + [(alg['name'], np.float) for alg in algorithms],
                      }
    else:
        _args_iter = {'object': [[tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_runtimes] * n_gen,
                      'dtype': [(alg['name'], np.float, (alg['n_iter'],)) for alg in algorithms],
                      }
        _args_mean = {'object': [[(np.nan,) * len(algorithms)] * n_runtimes] * n_gen,
                      'dtype': [(alg['name'], np.float) for alg in algorithms],
                      }

    l_ex_iter = np.array(**_args_iter)
    # t_run_iter = np.array(**_args_iter)

    l_ex_mean = np.array(**_args_mean)
    # t_run_mean = np.array(**_args_mean)

    # Generate scheduling problems
    for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose >= 1, save, file)):
        if solve:
            name = 'B&B Optimal'
            (tasks, ch_avail), (t_ex_opt, ch_ex_opt, t_run_opt) = out_gen

            check_valid(tasks, t_ex_opt, ch_ex_opt)
            l_ex_opt = eval_loss(tasks, t_ex_opt)

            l_ex_iter[name][i_gen, :, 0] = l_ex_opt
            # t_run_iter[name][i_gen, :, 0] = t_run_opt

            l_ex_mean[name][i_gen, :] = l_ex_opt
            # t_run_mean[name][i_gen, :] = t_run_opt

            # if verbose >= 2:       # TODO
            #     print(f'  {name}', end='\n')
            #     print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.2f}")
            #     # print(f"    Avg. Runtime: {t_run_mean[name][i_gen]:.2f} (s)")

        else:
            tasks, ch_avail = out_gen

        for name, func, n_iter in algorithms:
            if verbose >= 2:
                print(f'  {name}', end='\n')

            # for (i_time, runtime), iter_ in product(enumerate(runtimes), range(n_iter)):
            for iter_ in range(n_iter):      # Perform new algorithm runs

                if verbose >= 3:
                    # print(f'    Runtime: {i_time + 1}/{n_runtimes}, Iteration: {iter_ + 1}/{n_iter}', end='\r')
                    print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')      # TODO: combine above

                # Run algorithm
                solutions = list(func(tasks, ch_avail, runtimes))

                # t_ex, ch_ex = func(tasks, ch_avail, runtime)  # TODO: single call, yields! timeout wrapper?
                # t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)

                # Evaluate schedule
                for i_time, (t_ex, ch_ex) in enumerate(solutions):
                    check_valid(tasks, t_ex, ch_ex)
                    l_ex = eval_loss(tasks, t_ex)

                    # Store loss and runtime
                    l_ex_iter[name][i_gen, i_time, iter_] = l_ex
                    # l_ex_iter[name][i_gen, iter_] = l_ex
                    # t_run_iter[name][i_gen, iter_] = t_run

                    if plotting >= 3:
                        plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

            l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean(-1)
            # l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean()
            # t_run_mean[name][i_gen] = t_run_iter[name][i_gen].mean()

            # if verbose >= 2:       # TODO
            #     print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.2f}")
            #     # print(f"    Avg. Runtime: {t_run_mean[name][i_gen]:.2f} (s)")

        if plotting >= 2:
            _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
            plot_task_losses(tasks, ax=ax_gen[0])
            plot_loss_runtime(runtimes, l_ex_iter[i_gen], do_std=False, ax=ax_gen[1])  # TODO: just do scatter?
            # scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

    # Results
    if plotting >= 1:
        _, ax_results = plt.subplots(num='Results', clear=True)
        plot_loss_runtime(runtimes, l_ex_mean.transpose(), do_std=False,
                          ax=ax_results, ax_kwargs={'title': f'Performance on sets of {problem_gen.n_tasks} tasks'})

    # if solve:   # Relative to B&B     # TODO: normalized results?!?
    #     l_ex_mean_opt = l_ex_mean['B&B Optimal'].copy()
    #     # t_run_mean_opt = t_run_mean['B&B Optimal'].copy()
    #
    #     l_ex_mean_norm = l_ex_mean.copy()
    #     t_run_mean_norm = t_run_mean.copy()
    #
    #     l_ex_mean_norm['B&B Optimal'] = 0.
    #     # t_run_mean_norm['B&B Optimal'] = 0.
    #     for name in algorithms['name']:
    #         l_ex_mean_norm[name] -= l_ex_mean_opt
    #         l_ex_mean_norm[name] /= l_ex_mean_opt
    #         # t_run_mean_norm[name] -= t_run_mean_opt
    #         # t_run_mean_norm[name] /= t_run_mean_opt
    #
    #     if plotting >= 1:
    #         _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
    #         scatter_loss_runtime(t_run_mean_norm, l_ex_mean_norm,
    #                              ax=ax_results_norm,
    #                              ax_kwargs={'title': f'Relative Performance on sets '
    #                                                  f'of {problem_gen.n_tasks} tasks',
    #                                         'ylabel': 'Excess Loss (Normalized)',
    #                                         # 'xlabel': 'Runtime Difference (Normalized)'
    #                                         }
    #                              )

    return l_ex_iter
    # return l_ex_iter, t_run_iter


# TODO
# def compare_algorithms_lim(algorithms, runtimes, problem_gen, n_gen=1, solve=False, verbose=0, plotting=0, save=False, file=None):
#     n_runtimes = len(runtimes)
#
#     if solve:
#         _args_iter = {'object': [[([np.nan],) + tuple([np.nan] * alg['n_iter']
#                                                       for alg in algorithms)] * n_runtimes] * n_gen,
#                       'dtype': [(alg['name'], np.float, (alg['n_iter'],)) for alg in algorithms],
#                       }
#         _args_mean = {'object': [[(np.nan,) * (1 + len(algorithms))] * n_runtimes] * n_gen,
#                       'dtype': [('B&B Optimal', np.float)] + [(alg['name'], np.float) for alg in algorithms],
#                       }
#     else:
#         _args_iter = {'object': [[tuple([np.nan] * alg['n_iter'] for alg in algorithms)] * n_runtimes] * n_gen,
#                       'dtype': [(alg['name'], np.float, (alg['n_iter'],)) for alg in algorithms],
#                       }
#         _args_mean = {'object': [[(np.nan,) * len(algorithms)] * n_runtimes] * n_gen,
#                       'dtype': [(alg['name'], np.float) for alg in algorithms],
#                       }
#
#     l_ex_iter = np.array(**_args_iter)
#     # t_run_iter = np.array(**_args_iter)
#
#     l_ex_mean = np.array(**_args_mean)
#     # t_run_mean = np.array(**_args_mean)
#
#     # Generate scheduling problems
#     for i_gen, out_gen in enumerate(problem_gen(n_gen, solve, verbose >= 1, save, file)):
#         if solve:
#             (tasks, ch_avail), (t_ex_opt, ch_ex_opt, t_run_opt) = out_gen
#
#             check_valid(tasks, t_ex_opt, ch_ex_opt)
#             l_ex_opt = eval_loss(tasks, t_ex_opt)
#
#             l_ex_iter['B&B Optimal'][i_gen, 0] = l_ex_opt
#             # t_run_iter['B&B Optimal'][i_gen, 0] = t_run_opt
#
#             l_ex_mean['B&B Optimal'][i_gen] = l_ex_opt
#             # t_run_mean['B&B Optimal'][i_gen] = t_run_opt
#
#             if verbose >= 2:
#                 print(f'  B&B Optimal', end='\n')
#                 print(f"    Avg. Loss: {l_ex_mean['B&B Optimal'][i_gen]:.2f}")
#                 # print(f"    Avg. Runtime: {t_run_mean['B&B Optimal'][i_gen]:.2f} (s)")
#
#         else:
#             tasks, ch_avail = out_gen
#
#         for name, func, n_iter in algorithms:
#             if verbose >= 2:
#                 print(f'  {name}', end='\n')
#
#             for (i_time, runtime), iter_ in product(enumerate(runtimes), range(n_iter)):
#                 # for iter_ in range(n_iter):      # Perform new algorithm runs
#
#                 if verbose >= 3:
#                     print(f'    Runtime: {i_time + 1}/{n_runtimes}, Iteration: {iter_ + 1}/{n_iter}', end='\r')
#                     # print(f'    Iteration: {iter_ + 1}/{n_iter}', end='\r')
#
#                 # Run algorithm
#                 t_ex, ch_ex = func(tasks, ch_avail, runtime)  # TODO: single call, yields! timeout wrapper?
#                 # t_ex, ch_ex, t_run = timing_wrapper(func)(tasks, ch_avail)
#
#                 # Evaluate schedule
#                 check_valid(tasks, t_ex, ch_ex)
#                 l_ex = eval_loss(tasks, t_ex)
#
#                 # Store loss and runtime
#                 l_ex_iter[name][i_gen, i_time, iter_] = l_ex
#                 # l_ex_iter[name][i_gen, iter_] = l_ex
#                 # t_run_iter[name][i_gen, iter_] = t_run
#
#                 if plotting >= 3:
#                     plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)
#
#             l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean(-1)
#             # l_ex_mean[name][i_gen] = l_ex_iter[name][i_gen].mean()
#             # t_run_mean[name][i_gen] = t_run_iter[name][i_gen].mean()
#
#             if verbose >= 2:
#                 print(f"    Avg. Loss: {l_ex_mean[name][i_gen]:.2f}")
#                 # print(f"    Avg. Runtime: {t_run_mean[name][i_gen]:.2f} (s)")
#
#         if plotting >= 2:
#             _, ax_gen = plt.subplots(2, 1, num=f'Scheduling Problem: {i_gen + 1}', clear=True)
#             plot_task_losses(tasks, ax=ax_gen[0])
#             plot_loss_runtime(runtimes, l_ex_iter[i_gen], do_std=False, ax=ax_gen[1])  # TODO: just do scatter?
#             # scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])
#
#     # Results
#     if plotting >= 1:
#         _, ax_results = plt.subplots(num='Results', clear=True)
#         plot_loss_runtime(runtimes, l_ex_mean.transpose(), do_std=False,
#                           ax=ax_results, ax_kwargs={'title': f'Performance on sets of {problem_gen.n_tasks} tasks'})
#
#     # if solve:   # Relative to B&B     # TODO: normalized results?!?
#     #     l_ex_mean_opt = l_ex_mean['B&B Optimal'].copy()
#     #     # t_run_mean_opt = t_run_mean['B&B Optimal'].copy()
#     #
#     #     l_ex_mean_norm = l_ex_mean.copy()
#     #     t_run_mean_norm = t_run_mean.copy()
#     #
#     #     l_ex_mean_norm['B&B Optimal'] = 0.
#     #     # t_run_mean_norm['B&B Optimal'] = 0.
#     #     for name in algorithms['name']:
#     #         l_ex_mean_norm[name] -= l_ex_mean_opt
#     #         l_ex_mean_norm[name] /= l_ex_mean_opt
#     #         # t_run_mean_norm[name] -= t_run_mean_opt
#     #         # t_run_mean_norm[name] /= t_run_mean_opt
#     #
#     #     if plotting >= 1:
#     #         _, ax_results_norm = plt.subplots(num='Results (Normalized)', clear=True)
#     #         scatter_loss_runtime(t_run_mean_norm, l_ex_mean_norm,
#     #                              ax=ax_results_norm,
#     #                              ax_kwargs={'title': f'Relative Performance on sets '
#     #                                                  f'of {problem_gen.n_tasks} tasks',
#     #                                         'ylabel': 'Excess Loss (Normalized)',
#     #                                         # 'xlabel': 'Runtime Difference (Normalized)'
#     #                                         }
#     #                              )
#
#     return l_ex_iter
#     # return l_ex_iter, t_run_iter

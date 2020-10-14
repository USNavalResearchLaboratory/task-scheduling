"""
Task scheduler comparison.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from util.results import check_valid, eval_loss, timing_wrapper
from util.plot import plot_task_losses, scatter_loss_runtime, plot_schedule

from generators.scheduling_problems import Random as RandomProblem

from tree_search import TreeNodeShift, mcts, earliest_release
from Adam.env_tasking import StepTaskingEnv, train_agent, load_agent, SeqTaskingEnv, load_agent_sb, train_agent_sb ###### new includes

#################### new includes ##################################
####################################################################


# TODO: structure imports properly w/o relying on PyCharm's path append of the content root

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# logging.basicConfig(level=logging.INFO,       # TODO: logging?
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')


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
    file: str, optional
        File location relative to ../data/schedules/

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
                             ax_kwargs={'title': f'Performance on random sets of {problem_gen.n_tasks} tasks'})

    if verbose >= 1:
        print('\nAvg. Performance\n' + 16*'-')
        print(f"{'Algorithm:':<35}{'Loss:':<8}{'Runtime (s):':<10}")
        if solve:
            print(f"{'B&B Optimal':<35}{l_ex_mean['B&B Optimal'].mean():<8.2f}{t_run_mean['B&B Optimal'].mean():<10.6f}")
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
                                 ax_kwargs={'title': f'Relative Performance on random sets of {problem_gen.n_tasks} tasks',
                                            'ylabel': 'Excess Loss (Normalized)',
                                            # 'xlabel': 'Runtime Difference (Normalized)'
                                            }
                                 )

    return l_ex_iter, t_run_iter


def main():
    problem_gen = RandomProblem.relu_drop(n_tasks=6, n_ch=1)
    # problem_gen = ProblemDataset.load('temp/2020-08-26_14-36-22', iter_mode='once', shuffle=True, rng=None)

    # TODO: ensure train/test separation for loaded data, use iter_mode='once'
    # TODO: to train multiple schedulers on same loaded data, use problem_gen.restart() and problem_gen.shuffle = False?

    # Algorithms
    features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                         ('release time', lambda task: task.t_release,
                          (0., problem_gen.task_gen.param_lims['t_release'][1])),
                         ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                         ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                         ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                         ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.node.tasks[n].t_release

    env_cls = StepTaskingEnv
    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': True,
#                  'action_type': 'int',
                  'seq_encoding': 'one-hot',
                  }

    agent_file = None
    # agent_file = 'temp/2020-08-21_16-23-33'

    if agent_file is None:
        random_agent = train_agent(problem_gen, n_batch_train=3, n_batch_val=2, batch_size=1,
                                   env_cls=env_cls, env_params=env_params,
                                   save=True, save_dir=None)
    elif type(agent_file) == str:
        random_agent = load_agent(agent_file)
    else:
        raise ValueError("Parameter 'agent_file' must be string or None.")

################################ Stable Baselines Agent ####################################

#    env_cls = StepTaskingEnv
    sb_env_cls = SeqTaskingEnv
    sb_env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
#                  'masking': True,
                  'masking': False,
                  'action_type': 'int',
#                  'seq_encoding': 'one-hot',
                  }
    sb_env = SeqTaskingEnv(problem_gen, **sb_env_params)
    sb_agent_file = None
#    sb_agent_file = '../agents/DQNmlp_1000_steps.zip'

    if sb_agent_file is None:
        sb_agent = train_agent_sb(problem_gen, n_batch_train=3, n_batch_val=2, batch_size=1,
                                   env_cls=sb_env_cls, env_params=sb_env_params,
                                   save=True, save_dir=None, algorithm='DQN', policy='mlp',timesteps=20000)
    elif type(sb_agent_file) == str:
        sb_agent = load_agent_sb(sb_agent_file, sb_env)
    else:
        raise ValueError("Parameter 'agent_file' must be string or None.")
###############################################################################################

#    model_file = None
    # model_file = 'temp/2020-08-03_12-52-22'

#    if model_file is None:
#        network_policy = train_policy(problem_gen, n_batch_train=10, n_batch_val=10, batch_size=2,
#                                      env_cls=env_cls, env_params=env_params,
#                                      model=None, compile_params=None, fit_params=None,
#                                      do_tensorboard=False, plot_history=True, save=True, save_path=None)
#    elif type(model_file) == str:
#        network_policy = load_policy(model_file)
#    else:
#        raise ValueError("Parameter 'agent_file' must be string or None.")

    algorithms = np.array([
        # ('B&B', partial(branch_bound, verbose=False), 1),
        ('MCTS', partial(mcts, n_mc=200, verbose=False), 5),
        ('ERT', partial(earliest_release, do_swap=True), 1),
        ('Random Agent', partial(random_agent), 20),
        ('SB Agent', partial(sb_agent), 20),
#        ('DNN Policy', partial(network_policy), 1),
    ], dtype=[('name', '<U16'), ('func', object), ('n_iter', int)])

    # Compare algorithms
    compare_algorithms(algorithms, problem_gen, n_gen=10, solve=True, verbose=1, plotting=1, save=True, file=None)


if __name__ == '__main__':
    main()

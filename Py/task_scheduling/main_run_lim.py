"""
Task scheduler comparison with restricted algorithm runtime.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from util.generic import algorithm_repr
from util.results import check_valid, eval_loss
from util.plot import plot_task_losses, plot_loss_runtime

from generators.tasks import ContinuousUniformIID as ContinuousUniformTaskGenerator
from tree_search_run_lim import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release

plt.style.use('seaborn')


# TODO: INTEGRATE CHANGES from main.py!!!!


# %% Inputs
n_gen = 2      # number of task scheduling problems

n_tasks = 6
n_channels = 2

task_gen = ContinuousUniformTaskGenerator.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                                    t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)       # task set generator


max_runtimes = np.logspace(-2, 0, 11)

# Algorithms

# env = StepTaskingEnv(n_tasks, task_gen, n_channels, ch_avail_gen)
# random_agent = wrap_agent_run_lim(env, RandomAgent(env.action_space))


alg_funcs = [partial(branch_bound, verbose=False),
             partial(mcts_orig, n_mc=None, verbose=False),
             partial(mcts, verbose=False),
             partial(earliest_release, do_swap=True),
             partial(random_sequencer),
             # partial(random_agent)
             ]

alg_n_iters = [5, 5, 1, 1, 20]       # number of runs per problem

alg_reprs = list(map(algorithm_repr, alg_funcs))    # string representations


# %% Evaluate
n_runtimes = len(max_runtimes)

# l_ex_iter = np.array(list(zip(*[np.empty((n_gen*n_runtimes, n_iter)) for n_iter in alg_n_iters])),
#                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float],
#                                     list(zip(alg_n_iters))))).reshape(n_gen, n_runtimes)

# l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen*n_runtimes)))),
#                      dtype=[(alg_repr, np.float) for alg_repr in alg_reprs]).reshape(n_gen, n_runtimes)

l_ex_iter = np.array([[tuple([np.nan] * n_iter for n_iter in alg_n_iters)] * n_runtimes] * n_gen,
                     dtype=[(alg_repr, np.float, (n_iter,)) for alg_repr, n_iter in zip(alg_reprs, alg_n_iters)])

l_ex_mean = np.array([[(np.nan,) * len(alg_reprs)] * n_runtimes] * n_gen,
                     dtype=[(alg_repr, np.float) for alg_repr in alg_reprs])


for i_gen in range(n_gen):      # Generate new scheduling problem
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_iter in zip(alg_reprs, alg_funcs, alg_n_iters):
        # Perform new algorithm runs
        for (i_runtime, max_runtime), iter_ in product(enumerate(max_runtimes), range(n_iter)):
            print(f'  {alg_repr} - Runtime: {i_runtime + 1}/{n_runtimes} - Iteration: {iter_ + 1}/{n_iter}', end='\r')

            # Run algorithm
            t_ex, ch_ex = alg_func(tasks, ch_avail, max_runtime)

            # TODO: try/except, return NaN for algorithm timeout?

            # Evaluate schedule
            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            # Store loss
            l_ex_iter[alg_repr][i_gen, i_runtime, iter_] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

        l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean(-1)

    plot_loss_runtime(max_runtimes, l_ex_iter[i_gen], do_std=False, ax=ax_gen[1])

print('')

# Average results across random task sets
_, ax_results = plt.subplots(num='Results', clear=True)

plot_loss_runtime(max_runtimes, l_ex_mean.transpose(), do_std=False,
                  ax=ax_results, ax_kwargs={'title': 'Average performance on random task sets'})

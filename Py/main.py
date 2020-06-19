"""
Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

# TODO: Account for algorithm runtime before evaluating execution loss!!

# TODO: limit execution time of algorithms using signal module?
# TODO: add proper main() def, __name__ conditional execution?

import time     # TODO: use builtin module timeit instead? or cProfile?
from math import factorial, floor
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from util.generic import algorithm_repr, check_rng
from util.results import check_valid, eval_loss
from util.plot import plot_task_losses, plot_schedule, plot_results

from tasks import ReluDropGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release, est_alg_kw
from env_tasking import SeqTaskingEnv, StepTaskingEnv, wrap_agent, RandomAgent

plt.style.use('seaborn')


# %% Inputs
n_gen = 1      # number of task scheduling problems

n_tasks = 4
n_channels = 2

task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=None)       # task set generator


def ch_avail_gen(n_ch, rng=check_rng(None)):     # channel availability time generator
    # TODO: rng is a mutable default argument!
    return rng.uniform(0, 2, n_ch)


# Algorithms

env = StepTaskingEnv(n_tasks, task_gen, n_channels, ch_avail_gen)
random_agent = wrap_agent(env, RandomAgent(env.action_space))

alg_funcs = [partial(branch_bound, verbose=False),
             partial(mcts_orig, n_mc=[floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)], verbose=False),
             partial(mcts, verbose=False),
             partial(earliest_release, do_swap=True),
             partial(random_sequencer),
             partial(random_agent)]

alg_n_runs = [2, 2, 2, 1, 5, 5]       # number of runs per problem

alg_reprs = list(map(algorithm_repr, alg_funcs))


# %% Evaluate
t_run_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

l_ex_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

t_run_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

for i_gen in range(n_gen):      # Generate new tasks
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    tasks = task_gen.rand_tasks(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
        for i_run in range(n_run):      # Perform new algorithm runs
            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex = alg_func(tasks, ch_avail)
            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            t_run_iter[alg_repr][i_gen, i_run] = t_run
            l_ex_iter[alg_repr][i_gen, i_run] = l_ex

            plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

        t_run_mean[alg_repr][i_gen] = t_run_iter[alg_repr][i_gen].mean()
        l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean()

        print('')
        print(f"    Avg. Runtime: {t_run_mean[alg_repr][i_gen]:.2f} (s)")
        print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][i_gen]:.2f}")

    plot_results(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

print('')

# _, ax_results = plt.subplots(num='Results', clear=True)
# plot_results(t_run_mean, l_ex_mean, ax=ax_results, ax_kwargs={'title': 'Average performance on random task sets'})



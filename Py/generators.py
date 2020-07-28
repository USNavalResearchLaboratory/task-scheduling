"""Generator objects for tasks, channel availabilities, and complete tasking problems with optimal solutions."""

import time
import dill

import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_rng

from tasks import ReluDropTask
from tree_search import branch_bound

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# TODO: make channel availability generators for access to rng limits?


# Task generators        # TODO: generalize, add docstrings
class GenericTaskGenerator:
    def __init__(self, rng=None):
        self.rng = check_rng(rng)

    def __call__(self, n_tasks):
        raise NotImplementedError


class DeterministicTaskGenerator(GenericTaskGenerator):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.tasks


class PermuteTaskGenerator(GenericTaskGenerator):
    def __init__(self, tasks, rng=None):
        super().__init__(rng)
        self.tasks = tasks      # list of tasks

    def __call__(self, n_tasks):
        return self.rng.permutation(self.tasks)


class ReluDropGenerator(GenericTaskGenerator):
    """
    Generator of random ReluDropTask objects.

    Parameters
    ----------
    duration_lim : iterable of float
    t_release_lim : iterable of float
    slope_lim : iterable of float
    t_drop_lim : iterable of float
    l_drop_lim : iterable of float

    """

    def __init__(self, duration_lim, t_release_lim, slope_lim, t_drop_lim, l_drop_lim, rng=None):
        super().__init__(rng)
        self.duration_lim = duration_lim
        self.t_release_lim = t_release_lim
        self.slope_lim = slope_lim
        self.t_drop_lim = t_drop_lim
        self.l_drop_lim = l_drop_lim

    def __call__(self, n_tasks):
        """Randomly generate a list of tasks."""

        # TODO: generalize rng usage for non-uniform?

        duration = self.rng.uniform(*self.duration_lim, n_tasks)
        t_release = self.rng.uniform(*self.t_release_lim, n_tasks)
        slope = self.rng.uniform(*self.slope_lim, n_tasks)
        t_drop = self.rng.uniform(*self.t_drop_lim, n_tasks)
        l_drop = self.rng.uniform(*self.l_drop_lim, n_tasks)

        return [ReluDropTask(*args) for args in zip(duration, t_release, slope, t_drop, l_drop)]

        # for _ in range(n_tasks):      # FIXME: use yield?
        #     yield ReluDropTask(self.rng.uniform(*self.duration_lim),
        #                        self.rng.uniform(*self.t_release_lim),
        #                        self.rng.uniform(*self.slope_lim),
        #                        self.rng.uniform(*self.t_drop_lim),
        #                        self.rng.uniform(*self.l_drop_lim),
        #                        )

    @property
    def param_rep_lim(self):
        """Low and high tuples bounding parametric task representations."""
        return zip(self.duration_lim, self.t_release_lim, self.slope_lim, self.t_drop_lim, self.l_drop_lim)


# Tasking problem and solution generators
def schedule_gen(n_tasks, task_gen, n_ch, ch_avail_gen, n_gen=0, save=False, file_dir=None):
    """
    Generate optimal schedules for randomly generated tasking problems.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    task_gen : GenericTaskGenerator
        Task generation object.
    n_ch: int
        Number of channels.
    ch_avail_gen : callable
        Returns random initial channel availabilities.
    n_gen : int
        Number of tasking problems to generate.
    save : bool
        If True, the tasking problems and optimal schedules are serialized.
    file_dir : str, optional
        String representation of sub-directory to load from and/or save to.

    Returns
    -------
    dict
        Tasking problem generators and lists of tasking problems and their optimal schedules.

    """

    # TODO: train using complete tree info, not just B&B solution?
    # TODO: gitignore for ./data/ temps?
    # TODO: move to tasks.py

    dict_gen = {'n_tasks': n_tasks, 'task_gen': task_gen,
                'n_ch': n_ch, 'ch_avail_gen': ch_avail_gen,
                'tasks': [], 'ch_avail': [],
                't_ex': [], 'ch_ex': [],
                }

    # Search for existing file
    if file_dir is not None:
        try:
            with open('./data/schedules/' + file_dir, 'rb') as file:
                dict_gen_load = dill.load(file)

            # TODO: check equivalence of generators?

            print('File already exists. Appending new data.')
            dict_gen.update(dict_gen_load)

        except FileNotFoundError:
            pass

    # Generate tasks and find optimal schedules
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        tasks = task_gen(n_tasks)
        ch_avail = ch_avail_gen(n_ch)

        t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)   # optimal scheduler

        dict_gen['tasks'].append(tasks)
        dict_gen['ch_avail'].append(ch_avail)
        dict_gen['t_ex'].append(t_ex)
        dict_gen['ch_ex'].append(ch_ex)

        # TODO: yield??

    # Save schedules
    if save:
        if file_dir is None:
            file_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('./data/schedules/' + file_dir, 'wb') as file:
            dill.dump(dict_gen, file)    # save schedules

    return dict_gen


# FIXME
class SchedulingProblemGenerator:
    def __init__(self, n_tasks, task_gen, n_ch, ch_avail_gen):
        self.n_tasks = n_tasks
        self.task_gen = task_gen
        self.n_ch = n_ch
        self.ch_avail_gen = ch_avail_gen

    def __call__(self, n_gen):
        for _ in range(n_gen):

            tasks = self.task_gen(self.n_tasks)
            ch_avail = self.ch_avail_gen(self.n_ch)

            yield tasks, ch_avail

            t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)  # optimal scheduler


class LoadProblemGenerator:
    def __init__(self):
        pass

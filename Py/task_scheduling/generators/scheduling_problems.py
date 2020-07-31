"""Generator objects for tasks, channel availabilities, and complete tasking problems with optimal solutions."""

import time
import dill
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng

from generators.tasks import ReluDrop as ReluDropGenerator
from generators.channel_availabilities import Uniform as UniformChanGenerator

from tree_search import branch_bound

np.set_printoptions(precision=2)
plt.style.use('seaborn')


# Tasking problem and solution generators
def schedule_gen(n_tasks, task_gen, n_ch, ch_avail_gen, n_gen=0, save=False, file=None):
    """
    Generate optimal schedules for randomly generated tasking problems.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    task_gen : generators.tasks.Base
        Task generation object.
    n_ch: int
        Number of channels.
    ch_avail_gen : callable
        Returns random initial channel availabilities.
    n_gen : int
        Number of tasking problems to generate.
    save : bool
        If True, the tasking problems and optimal schedules are serialized.
    file : str, optional
        String representation of file path (relative to module root) to load from and/or save to.

    Returns
    -------
    dict
        Tasking problem generators and lists of tasking problems and their optimal schedules.

    """

    # FIXME: delete?

    dict_gen = {'n_tasks': n_tasks, 'task_gen': task_gen,
                'n_ch': n_ch, 'ch_avail_gen': ch_avail_gen,
                'tasks': [], 'ch_avail': [],
                't_ex': [], 'ch_ex': [],
                }

    # Search for existing file
    if file is not None:
        try:
            with open('../../data/schedules/' + file, 'rb') as file:
                dict_gen_load = dill.load(file)

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

    # Save schedules
    if save:
        if file is None:
            file = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('../../data/schedules/' + file, 'wb') as file:
            dill.dump(dict_gen, file)    # save schedules

    return dict_gen


# FIXME: WIP
class Base:
    def __init__(self, n_tasks, n_ch):
        self.n_tasks = n_tasks
        self.n_ch = n_ch

        self._SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])


class Random(Base):
    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen):
        super().__init__(n_tasks, n_ch)
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

    @classmethod
    def relu_drop_default(cls, n_tasks, n_ch):
        task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                     t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

        ch_avail_gen = UniformChanGenerator(lims=(0, 1))

        return cls(n_tasks, n_ch, task_gen, ch_avail_gen)

    def __call__(self, n_gen=1):
        for _ in range(n_gen):
            tasks = list(self.task_gen(self.n_tasks))
            ch_avail = list(self.ch_avail_gen(self.n_ch))

            yield self._SchedulingProblem(tasks, ch_avail)

    def __eq__(self, other):
        if not isinstance(other, Random):
            return False

        conditions = [self.n_tasks == other.n_tasks,
                      self.n_ch == other.n_ch,
                      self.task_gen == other.task_gen,
                      self.ch_avail_gen == other.ch_avail_gen]

        return True if all(conditions) else False

    def schedule_gen(self, n_gen, save=False, file=None):
        dict_gen = {'problem_gen': self,
                    # 'n_tasks': self.n_tasks, 'task_gen': self.task_gen,
                    # 'n_ch': self.n_ch, 'ch_avail_gen': self.ch_avail_gen,
                    'problems': [],
                    # 't_ex': [], 'ch_ex': [],
                    }

        # Search for existing file
        if file is not None:
            try:
                with open('../../data/schedules/' + file, 'rb') as file:
                    dict_gen_load = dill.load(file)

                # Check equivalence of generators
                if dict_gen_load['problem_gen'] == self:
                    print('File already exists. Appending new data.')
                    dict_gen.update(dict_gen_load)

            except FileNotFoundError:
                pass

        # Generate tasks and find optimal schedules
        for i_gen, problem in enumerate(self(n_gen)):
            print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

            dict_gen['problems'].append(problem)

            # TODO: train using complete tree info, not just B&B solution?
            # t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)  # optimal scheduler
            # dict_gen['t_ex'].append(t_ex)
            # dict_gen['ch_ex'].append(ch_ex)

            # yield problem     # TODO: yield?

        # Save schedules
        if save:
            if file is None:
                file = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

            with open('../../data/schedules/' + file, 'wb') as file:
                dill.dump(dict_gen, file)  # save schedules

        return dict_gen


class Load(Base):

    # FIXME: just serialize this object with the data stored in its attributes? or serialize the Random one?!?!?!

    def __init__(self, file, iter_mode='repeat', shuffle=True, rng=None):
        with open('../data/schedules/' + file, 'rb') as file:
            dict_gen = dill.load(file)

        problem_gen = dict_gen['problem_gen']

        super().__init__(problem_gen.n_tasks, problem_gen.n_ch)
        self.task_gen = problem_gen.task_gen
        self.ch_avail_gen = problem_gen.ch_avail_gen

        # super().__init__(dict_gen['n_tasks'], dict_gen['n_ch'])
        # self.task_gen = dict_gen['task_gen']
        # self.ch_avail_gen = dict_gen['ch_avail_gen']

        self.problems = dict_gen['problems']

        self.iter_mode = iter_mode
        self.shuffle = shuffle
        self.rng = check_rng(rng)

        self.i = None
        self.restart()

    def restart(self):
        self.i = 0
        if self.shuffle:
            self.rng.shuffle(self.problems)

    def __call__(self, n_gen=1):
        for _ in range(n_gen):
            if self.i == len(self.problems):
                if self.iter_mode == 'once':
                    warnings.warn("Problem generator data has been exhausted.")
                    return
                elif self.iter_mode == 'repeat':
                    self.i = 0

            problem = self.problems[self.i]

            yield problem
            self.i += 1


def main():
    rand_gen = Random.relu_drop_default(n_tasks=6, n_ch=2)

    probs = list(rand_gen(n_gen=3))

    p = list(rand_gen.schedule_gen(n_gen=3, save=True))

    # load_gen = Load('temp/2020-07-29_10-02-56')
    # probs2 = list(load_gen(n_gen=2))


if __name__ == '__main__':
    main()

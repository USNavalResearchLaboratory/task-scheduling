"""Generator objects for tasks, channel availabilities, and complete tasking problems with optimal solutions."""

import time
import dill
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng
from generators.tasks import ReluDrop as ReluDropTaskGenerator
from generators.channel_availabilities import Uniform as UniformChanGenerator
from tree_search import branch_bound

np.set_printoptions(precision=2)
plt.style.use('seaborn')


# Tasking problem and solution generators
class Base:
    def __init__(self, n_tasks, n_ch, rng=None):
        self.n_tasks = n_tasks
        self.n_ch = n_ch
        self.rng = check_rng(rng)


class Random(Base):
    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen, rng=None):
        super().__init__(n_tasks, n_ch, rng)
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

        self._SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])
        self._SchedulingSolution = namedtuple('SchedulingSolution', ['t_ex', 'ch_ex'])

    @classmethod
    def relu_drop_default(cls, n_tasks, n_ch, rng=None):
        _rng = check_rng(rng)

        task_gen = ReluDropTaskGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                         t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=_rng)  # task set generator

        ch_avail_gen = UniformChanGenerator(lim=(0, 1), rng=_rng)

        return cls(n_tasks, n_ch, task_gen, ch_avail_gen, _rng)

    def __eq__(self, other):
        if not isinstance(other, Random):
            return False

        conditions = [self.n_tasks == other.n_tasks,
                      self.n_ch == other.n_ch,
                      self.task_gen == other.task_gen,
                      self.ch_avail_gen == other.ch_avail_gen]

        return True if all(conditions) else False

    # def __call__(self, n_gen=1):
    #     for _ in range(n_gen):
    #         tasks = list(self.task_gen(self.n_tasks))
    #         ch_avail = list(self.ch_avail_gen(self.n_ch))
    #
    #         yield self._SchedulingProblem(tasks, ch_avail)

    def __call__(self, n_gen=1, solve=False, save=False, file=None):

        # FIXME: variable number of yielded arguments!? Modify problem_gen calls in train_'s and env.reset

        # FIXME: save optimal solutions
        # TODO: train using complete tree info, not just B&B solution?

        if save:
            dict_gen = {'problem_gen': self,
                        'problems': [],
                        }
            if solve:
                dict_gen.update(solutions=[])

        # Generate tasks and find optimal schedules
        for _ in range(n_gen):
            tasks = list(self.task_gen(self.n_tasks))
            ch_avail = list(self.ch_avail_gen(self.n_ch))

            problem = self._SchedulingProblem(tasks, ch_avail)
            if solve:
                t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=False)  # optimal scheduler
                solution = self._SchedulingSolution(t_ex, ch_ex)

                yield problem, solution
            else:
                yield problem

            if save:
                dict_gen['problems'].append(problem)
                if solve:
                    dict_gen['solutions'].append(solution)

        if save:
            if file is None:
                file = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                try:
                    with open('data/schedules/' + file, 'rb') as file:
                        dict_gen_load = dill.load(file)

                    # Check equivalence of generators
                    conditions = [dict_gen_load['problem_gen'] == self,
                                  ('solutions' in dict_gen_load.keys()) == solve    # both generators do or do not solve
                                  ]
                    if all(conditions):
                        print('File already exists. Appending existing data.')

                        dict_gen['problems'] += dict_gen_load['problems']
                        if solve:
                            dict_gen['solutions'] += dict_gen_load['solutions']

                except FileNotFoundError:
                    pass

            with open('data/schedules/' + file, 'wb') as file:
                dill.dump(dict_gen, file)  # save schedules


    # def schedule_gen(self, n_gen, save=False, file=None):
    #     dict_gen = {'problem_gen': self,
    #                 'problems': [],
    #                 # 't_ex': [], 'ch_ex': [],
    #                 }
    #
    #     # Search for existing file
    #     if file is not None:
    #         try:
    #             with open('data/schedules/' + file, 'rb') as file:
    #                 dict_gen_load = dill.load(file)
    #
    #             # Check equivalence of generators
    #             if dict_gen_load['problem_gen'] == self:
    #                 print('File already exists. Appending new data.')
    #                 dict_gen = dict_gen_load
    #
    #         except FileNotFoundError:
    #             pass
    #
    #     # Generate tasks and find optimal schedules
    #     for i_gen, problem in enumerate(self(n_gen)):
    #         print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')
    #
    #         dict_gen['problems'].append(problem)
    #
    #         # FIXME: save optimal solutions
    #         # TODO: train using complete tree info, not just B&B solution?
    #
    #         # t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)  # optimal scheduler
    #         # dict_gen['t_ex'].append(t_ex)
    #         # dict_gen['ch_ex'].append(ch_ex)/**
    #
    #     # Save schedules
    #     if save:
    #         if file is None:
    #             file = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
    #
    #         with open('data/schedules/' + file, 'wb') as file:
    #             dill.dump(dict_gen, file)  # save schedules
    #
    #     return dict_gen['problems']


class Dataset(Base):
    def __init__(self, problems, problem_gen, iter_mode='repeat', shuffle=True, rng=None):
        self.problems = problems

        # TODO: these attributes only needed for Env observation space lims
        super().__init__(problem_gen.n_tasks, problem_gen.n_ch, rng)
        self.task_gen = problem_gen.task_gen
        self.ch_avail_gen = problem_gen.ch_avail_gen

        self.iter_mode = iter_mode
        self.shuffle = shuffle

        self.i = None
        self.restart()

    @classmethod
    def load(cls, file, iter_mode='repeat', shuffle=True, rng=None):
        with open('data/schedules/' + file, 'rb') as file:
            dict_gen = dill.load(file)

        return cls(dict_gen['problems'], dict_gen['problem_gen'], iter_mode, shuffle, rng)

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
    rand_gen = Random.relu_drop_default(n_tasks=3, n_ch=2, rng=None)

    print(list(rand_gen(2)))
    print(list(rand_gen(3)))

    # load_gen = Dataset('temp/2020-07-29_10-02-56')
    # probs2 = list(load_gen(n_gen=2))


if __name__ == '__main__':
    main()

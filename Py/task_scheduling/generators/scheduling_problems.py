"""Generator objects for complete tasking problems with optimal solutions."""

from time import strftime
import dill
import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng
from util.results import timing_wrapper, check_valid
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

        self._SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])
        self._SchedulingSolution = namedtuple('SchedulingSolution', ['t_ex', 'ch_ex', 't_run'], defaults=(None,))

    # TODO: DRY principle - move call code here?

    @staticmethod
    def save(pkl_dict, file=None):
        if file is None:
            file = 'temp/{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            try:
                with open('data/schedules/' + file, 'rb') as file:
                    pkl_dict_load = dill.load(file)

                # Check equivalence of generators
                if pkl_dict_load['problem_gen'] == pkl_dict['problem_gen']:
                    print('File already exists. Appending existing data.')

                    pkl_dict['problems'] += pkl_dict_load['problems']
                    if 'solutions' in pkl_dict.keys():
                        try:
                            pkl_dict['solutions'] += pkl_dict_load['solutions']
                        except KeyError:
                            return

            except FileNotFoundError:
                pass

        with open('data/schedules/' + file, 'wb') as file:
            dill.dump(pkl_dict, file)  # save schedules


class Random(Base):
    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen, rng=None):
        super().__init__(n_tasks, n_ch, rng)
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

    @classmethod
    def relu_drop_default(cls, n_tasks, n_ch, rng=None):
        _rng = check_rng(rng)

        # task_gen = ReluDropTaskGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
        #                                  t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=_rng)  # task set generator
        task_gen = ReluDropTaskGenerator.iid_uniform(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                                     t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=_rng)

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

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):

        # TODO: buffer generated data across multiple calls, save and clear buffer using separate method?
        # TODO: train using complete tree info, not just B&B solution?

        pkl_dict = {'problem_gen': self, 'problems': []}
        if solve:
            pkl_dict.update(solutions=[])

        # Generate tasks and find optimal schedules
        for i_gen in range(n_gen):
            if verbose:
                print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

            tasks = list(self.task_gen(self.n_tasks))
            ch_avail = list(self.ch_avail_gen(self.n_ch))

            problem = self._SchedulingProblem(tasks, ch_avail)
            if save:
                pkl_dict['problems'].append(problem)

            if solve:
                t_ex, ch_ex, t_run = timing_wrapper(partial(branch_bound, verbose=verbose))(tasks, ch_avail)
                solution = self._SchedulingSolution(t_ex, ch_ex, t_run)
                if save:
                    pkl_dict['solutions'].append(solution)

                yield problem, solution
            else:
                yield problem

        if save:
            self.save(pkl_dict, file)
            # if file is None:
            #     file = 'temp/{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))
            # else:
            #     try:
            #         with open('data/schedules/' + file, 'rb') as file:
            #             pkl_dict_load = dill.load(file)
            #
            #         # Check equivalence of generators
            #         conditions = [pkl_dict_load['problem_gen'] == pkl_dict['problem_gen'],
            #                       ('solutions' in pkl_dict_load.keys()) == solve    # both generators do or do not solve
            #                       ]
            #         if all(conditions):
            #             print('File already exists. Appending existing data.')
            #
            #             pkl_dict['problems'] += pkl_dict_load['problems']
            #             if solve:
            #                 pkl_dict['solutions'] += pkl_dict_load['solutions']
            #
            #     except FileNotFoundError:
            #         pass
            #
            # with open('data/schedules/' + file, 'wb') as file:
            #     dill.dump(pkl_dict, file)  # save schedules


class Dataset(Base):
    def __init__(self, problem_gen, problems, solutions=None, iter_mode='repeat', shuffle=True, rng=None):

        # TODO: these attributes only needed for Env observation space lims
        super().__init__(problem_gen.n_tasks, problem_gen.n_ch, rng)
        self.problem_gen = problem_gen
        self.task_gen = problem_gen.task_gen
        self.ch_avail_gen = problem_gen.ch_avail_gen

        self.problems = problems
        self.solutions = solutions

        self.iter_mode = iter_mode
        self.shuffle = shuffle

        self.i = None
        self.restart()

    @classmethod
    def load(cls, file, iter_mode='repeat', shuffle=True, rng=None):
        # TODO: loading entire dict of data into attribute defeats purpose of generator yield!?

        with open('data/schedules/' + file, 'rb') as file:
            dict_gen = dill.load(file)

        solutions = dict_gen['solutions'] if 'solutions' in dict_gen.keys() else None
        return cls(dict_gen['problem_gen'], dict_gen['problems'], solutions, iter_mode, shuffle, rng)

    def restart(self):
        self.i = 0
        if self.shuffle:
            i_permute = self.rng.permutation(len(self.problems)).tolist()
            self.problems = [self.problems[i] for i in i_permute]
            if self.solutions is not None:
                self.solutions = [self.solutions[i] for i in i_permute]

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):
        pkl_dict = {'problem_gen': self.problem_gen, 'problems': []}
        if solve:
            pkl_dict.update(solutions=[])

        for i_gen in range(n_gen):
            if verbose:
                print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

            if self.i == len(self.problems):
                if self.iter_mode == 'once':
                    warnings.warn("Problem generator data has been exhausted.")
                    return
                elif self.iter_mode == 'repeat':
                    self.restart()

            problem = self.problems[self.i]
            if save:
                pkl_dict['problems'].append(problem)

            if solve:
                if self.solutions is None:
                    t_ex, ch_ex, t_run = timing_wrapper(partial(branch_bound, verbose=verbose))(*problem)
                    solution = self._SchedulingSolution(t_ex, ch_ex, t_run)
                else:
                    solution = self.solutions[self.i]

                if save:
                    pkl_dict['solutions'].append(solution)

                yield problem, solution
            else:
                yield problem

            self.i += 1

        if save:
            self.save(pkl_dict, file)
            # if file is None:
            #     file = 'temp/{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))
            # else:
            #     try:
            #         with open('data/schedules/' + file, 'rb') as file:
            #             pkl_dict_load = dill.load(file)
            #
            #         # Check equivalence of generators
            #         conditions = [pkl_dict_load['problem_gen'] == pkl_dict['problem_gen'],
            #                       ('solutions' in pkl_dict_load.keys()) == solve    # both generators do or do not solve
            #                       ]
            #         if all(conditions):
            #             print('File already exists. Appending existing data.')
            #
            #             pkl_dict['problems'] += pkl_dict_load['problems']
            #             if solve:
            #                 pkl_dict['solutions'] += pkl_dict_load['solutions']
            #
            #     except FileNotFoundError:
            #         pass
            #
            # with open('data/schedules/' + file, 'wb') as file:
            #     dill.dump(pkl_dict, file)  # save schedules


def main():
    rand_gen = Random.relu_drop_default(n_tasks=3, n_ch=2, rng=None)

    print(list(rand_gen(2)))
    print(list(rand_gen(3)))

    # load_gen = Dataset('temp/2020-07-29_10-02-56')
    # probs2 = list(load_gen(n_gen=2))


if __name__ == '__main__':
    main()

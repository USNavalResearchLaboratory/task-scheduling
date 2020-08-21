"""Generator objects for complete tasking problems with optimal solutions."""

from time import strftime
import dill
import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from util.generic import check_rng
from util.results import timing_wrapper
from generators.tasks import ContinuousUniformIID as ContinuousUniformTaskGenerator
from generators.channel_availabilities import UniformIID as UniformChanGenerator
from tree_search import branch_bound

np.set_printoptions(precision=2)
plt.style.use('seaborn')

# TODO: docstrings and comments


class Base:
    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen, rng=None):
        self.n_tasks = n_tasks
        self.n_ch = n_ch
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

        self.rng = check_rng(rng)

        self._SchedulingProblem = namedtuple('SchedulingProblem', ['tasks', 'ch_avail'])
        self._SchedulingSolution = namedtuple('SchedulingSolution', ['t_ex', 'ch_ex', 't_run'], defaults=(None,))

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):
        raise NotImplementedError   # TODO: DRY principle?

    # def gen_single(self, solve, verbose, save):   # TODO: delete?
    #     raise NotImplementedError

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

                    # Append loaded problems and solutions
                    if 'solutions' in pkl_dict.keys():
                        try:
                            pkl_dict['solutions'] += pkl_dict_load['solutions']
                            pkl_dict['problems'] += pkl_dict_load['problems']
                        except KeyError:
                            pass
                    else:
                        pkl_dict['problems'] += pkl_dict_load['problems']

            except FileNotFoundError:
                pass

        with open('data/schedules/' + file, 'wb') as file:
            dill.dump(pkl_dict, file)  # save schedules

    def __eq__(self, other):
        if not isinstance(other, Base):
            return False
        else:
            conditions = [self.n_tasks == other.n_tasks,
                          self.n_ch == other.n_ch,
                          self.task_gen == other.task_gen,
                          self.ch_avail_gen == other.ch_avail_gen]

            return True if all(conditions) else False


class Random(Base):

    @classmethod
    def relu_drop_default(cls, n_tasks, n_ch, rng=None):
        _rng = check_rng(rng)

        task_gen = ContinuousUniformTaskGenerator.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                                            t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=_rng)
        ch_avail_gen = UniformChanGenerator(lim=(0, 1), rng=_rng)

        return cls(n_tasks, n_ch, task_gen, ch_avail_gen, _rng)

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):

        # TODO: buffer generated data across multiple calls, save and clear buffer using separate method?

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


class Dataset(Base):
    def __init__(self, problem_gen, problems, solutions=None, iter_mode='repeat', shuffle=True, rng=None):
        super().__init__(problem_gen.n_tasks, problem_gen.n_ch, problem_gen.task_gen, problem_gen.ch_avail_gen, rng)
        self.problem_gen = problem_gen

        self.problems = problems
        self.solutions = solutions

        self.iter_mode = iter_mode
        self.shuffle = shuffle

        self.i = None
        self.restart()

    @classmethod
    def load(cls, file, iter_mode='repeat', shuffle=True, rng=None):
        # TODO: loading entire dict into attribute defeats purpose of generator yield!?

        with open('data/schedules/' + file, 'rb') as file:
            dict_gen = dill.load(file)

        solutions = dict_gen['solutions'] if 'solutions' in dict_gen.keys() else None
        return cls(dict_gen['problem_gen'], dict_gen['problems'], solutions, iter_mode, shuffle, rng)

    def restart(self):
        self.i = 0
        if self.shuffle:
            if self.solutions is None:
                self.problems = self.rng.permutation(self.problems).tolist()
            else:
                _temp = list(zip(self.problems, self.solutions))
                _p, _s = zip(*self.rng.permutation(_temp).tolist())
                self.problems, self.solutions = list(_p), list(_s)

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


def main():
    rand_gen = Random.relu_drop_default(n_tasks=3, n_ch=2, rng=None)

    print(list(rand_gen(2)))
    print(list(rand_gen(3)))

    # load_gen = Dataset('temp/2020-07-29_10-02-56')
    # probs2 = list(load_gen(n_gen=2))


if __name__ == '__main__':
    main()

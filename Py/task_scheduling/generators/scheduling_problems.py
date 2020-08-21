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
    """
    Base class for scheduling problem generators.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n_ch: int
        Number of channels.
    task_gen : generators.tasks.BaseIID, optional
        Task generation object.
    ch_avail_gen : generators.channel_availabilities.BaseIID, optional
        Returns random initial channel availabilities.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

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

    @staticmethod
    def save(save_dict, file=None):
        """
        Serialize scheduling problems/solutions.

        Parameters
        ----------
        save_dict: dict
            Serialized dict with keys 'problems', 'solutions', 'task_gen', and 'ch_avail_gen'.
        file : str, optional
            File location relative to ./data/schedules/

        """

        if file is None:
            file = 'temp/{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            try:    # search for existing file
                with open('data/schedules/' + file, 'rb') as file:
                    load_dict = dill.load(file)

                # Check equivalence of generators
                conditions = [load_dict['task_gen'] == save_dict['task_gen'],
                              load_dict['ch_avail_gen'] == save_dict['ch_avail_gen'],
                              len(load_dict['problems'][0].tasks) == len(save_dict['problems'][0].tasks),
                              len(load_dict['problems'][0].ch_avail) == len(save_dict['problems'][0].ch_avail),
                              ]

                if all(conditions):     # Append loaded problems and solutions
                    print('File already exists. Appending existing data.')

                    if 'solutions' in save_dict.keys():
                        try:
                            save_dict['solutions'] += load_dict['solutions']
                            save_dict['problems'] += load_dict['problems']
                        except KeyError:
                            pass    # Skip if new data has solutions and loaded data does not
                    else:
                        save_dict['problems'] += load_dict['problems']

            except FileNotFoundError:
                pass

        with open('data/schedules/' + file, 'wb') as file:
            dill.dump(save_dict, file)  # save schedules

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

        task_gen = ContinuousUniformTaskGenerator.relu_drop(duration_lim=(3, 6), t_release_lim=(0, 4),
                                                            slope_lim=(0.5, 2), t_drop_lim=(6, 12),
                                                            l_drop_lim=(35, 50), rng=_rng)
        ch_avail_gen = UniformChanGenerator(lim=(0, 1), rng=_rng)

        return cls(n_tasks, n_ch, task_gen, ch_avail_gen, _rng)

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):
        """
        Call problem generator.

        Parameters
        ----------
        n_gen : int, optional
            Number of problems to generate.
        solve : bool, optional
            Enables generation of Branch & Bound optimal solutions.
        verbose : bool, optional
            Enables print-out progress information.
        save : bool, optional
            Enables serialization of generated problems/solutions.
        file: str, optional
            File location relative to ./data/schedules/

        Yields
        ------
        SchedulingProblem or tuple
            If 'solve' is True, yields 2-tuple of (SchedulingProblem, SchedulingSolution)

        """

        # TODO: buffer generated data across multiple calls, save and clear buffer using separate method?

        save_dict = {'problems': [], 'solutions': [] if solve else None,
                     'task_gen': self.task_gen, 'ch_avail_gen': self.ch_avail_gen}

        # Generate tasks and find optimal schedules
        for i_gen in range(n_gen):
            if verbose:
                print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

            # Randomly generated problem
            tasks = list(self.task_gen(self.n_tasks))
            ch_avail = list(self.ch_avail_gen(self.n_ch))

            problem = self._SchedulingProblem(tasks, ch_avail)
            if save:
                save_dict['problems'].append(problem)

            if solve:
                # Generate B&B optimal solution
                t_ex, ch_ex, t_run = timing_wrapper(partial(branch_bound, verbose=verbose))(tasks, ch_avail)
                solution = self._SchedulingSolution(t_ex, ch_ex, t_run)
                if save:
                    save_dict['solutions'].append(solution)

                yield problem, solution
            else:
                yield problem

        if save:
            self.save(save_dict, file)


class Dataset(Base):
    """
    Generator of scheduling problems in memory.

    Parameters
    ----------
    problems : list of SchedulingProblem
        Scheduling problems
    solutions : list of SchedulingSolution, optional
        Optimal scheduling solutions
    task_gen : generators.tasks.BaseIID, optional
        Task generation object.
    ch_avail_gen : generators.channel_availabilities.BaseIID, optional
        Returns random initial channel availabilities.
    iter_mode : str, optional
        If 'once', generator call raises a warning when all data has been yielded. If 'repeat', a new pass is started.
    shuffle : bool, optional
        Enables uniformly random permutation of problems/solutions before generation.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, problems: list, solutions=None, task_gen=None, ch_avail_gen=None,
                 iter_mode='once', shuffle=True, rng=None):

        self.problems = problems
        self.solutions = solutions

        super().__init__(len(problems[0].tasks), len(problems[0].ch_avail), task_gen, ch_avail_gen, rng)

        self.iter_mode = iter_mode
        self.shuffle = shuffle

        self.i = None
        self.restart()

    @classmethod
    def load(cls, file, iter_mode='once', shuffle=True, rng=None):
        """Load problems/solutions from memory."""

        # TODO: loading entire dict into attribute defeats purpose of generator yield!?
        with open('data/schedules/' + file, 'rb') as file:
            dict_gen = dill.load(file)

        return cls(**dict_gen, iter_mode=iter_mode, shuffle=shuffle, rng=rng)

    def restart(self):
        """Resets data index pointer to beginning, performs optional data shuffle."""

        self.i = 0
        if self.shuffle:
            if self.solutions is None:
                self.problems = self.rng.permutation(self.problems).tolist()
            else:
                _temp = list(zip(self.problems, self.solutions))
                _p, _s = zip(*self.rng.permutation(_temp).tolist())
                self.problems, self.solutions = list(_p), list(_s)

    def __call__(self, n_gen=1, solve=False, verbose=False, save=False, file=None):
        """
        Call problem generator.

        Parameters
        ----------
        n_gen : int, optional
            Number of problems to generate.
        solve : bool, optional
            Enables generation of Branch & Bound optimal solutions.
        verbose : bool, optional
            Enables print-out progress information.
        save : bool, optional
            Enables serialization of generated problems/solutions.
        file: str, optional
            File location relative to ./data/schedules/

        Yields
        ------
        SchedulingProblem or tuple
            If 'solve' is True, yields 2-tuple of (SchedulingProblem, SchedulingSolution)

        """

        save_dict = {'problems': [], 'solutions': [] if solve else None,
                     'task_gen': self.task_gen, 'ch_avail_gen': self.ch_avail_gen}

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
                save_dict['problems'].append(problem)

            if solve:
                if self.solutions is None:
                    # Generate B&B optimal solution
                    t_ex, ch_ex, t_run = timing_wrapper(partial(branch_bound, verbose=verbose))(*problem)
                    solution = self._SchedulingSolution(t_ex, ch_ex, t_run)
                else:   # load solution
                    solution = self.solutions[self.i]

                if save:
                    save_dict['solutions'].append(solution)

                yield problem, solution
            else:
                yield problem

            self.i += 1

        if save:
            self.save(save_dict, file)


def main():
    rand_gen = Random.relu_drop_default(n_tasks=3, n_ch=2, rng=None)

    print(list(rand_gen(2)))
    print(list(rand_gen(3)))

    # load_gen = Dataset('temp/2020-07-29_10-02-56')
    # probs2 = list(load_gen(n_gen=2))


if __name__ == '__main__':
    main()

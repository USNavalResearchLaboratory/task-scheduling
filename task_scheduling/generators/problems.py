"""Generator objects for complete tasking problems with optimal solutions."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path

import dill
import numpy as np

from task_scheduling.algorithms import branch_bound_priority
from task_scheduling.base import RandomGeneratorMixin, SchedulingProblem, SchedulingSolution
from task_scheduling.generators import tasks as task_gens, channels as chan_gens
from task_scheduling.util import eval_wrapper


class Base(RandomGeneratorMixin, ABC):
    temp_path = None

    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen, rng=None):
        """
        Base class for scheduling problem generators.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        n_ch: int
            Number of channels.
        task_gen : generators.tasks.Base
            Task generation object.
        ch_avail_gen : generators.channels.Base
            Returns random initial channel availabilities.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        """

        super().__init__(rng)

        self.n_tasks = n_tasks
        self.n_ch = n_ch
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

    def __call__(self, n_gen, solve=False, verbose=0, save_path=None, rng=None):
        """
        Call problem generator.

        Parameters
        ----------
        n_gen : int
            Number of scheduling problems to generate.
        solve : bool, optional
            Enables generation of Branch & Bound optimal solutions.
        verbose : int, optional
            Progress print-out level. '0' is silent, '1' prints iteration number, '2' prints solver progress.
        save_path : os.PathLike or str, optional
            File path for saving data.
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Yields
        ------
        SchedulingProblem
            Scheduling problem namedtuple.
        SchedulingSolution, optional
            Scheduling solution namedtuple.

        """

        problems = []
        solutions = [] if solve else None

        if save_path is None and self.temp_path is not None:
            now = datetime.now().replace(microsecond=0).isoformat().replace(':', '_')
            save_path = Path(self.temp_path) / now

        save = save_path is not None

        # Generate tasks and find optimal schedules
        rng = self._get_rng(rng)
        for i_gen in range(n_gen):
            if verbose >= 1:
                end = '\r' if verbose == 1 else '\n'
                print(f'Problem: {i_gen + 1}/{n_gen}', end=end)

            problem = self._gen_problem(rng)
            if save:
                problems.append(problem)

            if solve:
                solution = self._gen_solution(problem, verbose >= 2)
                if save:
                    solutions.append(solution)

                yield problem, solution
            else:
                yield problem

        if save:
            self._save(problems, solutions, save_path)

    @abstractmethod
    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        raise NotImplementedError

    @staticmethod
    def _gen_solution(problem, verbose=False):
        # scheduler_opt = partial(branch_bound, verbose=verbose)
        scheduler_opt = partial(branch_bound_priority, verbose=verbose)

        return eval_wrapper(scheduler_opt)(*problem)

    def _save(self, problems, solutions=None, file_path=None):
        """
        Serialize scheduling problems/solutions.

        Parameters
        ----------
        problems : Collection of SchedulingProblem
            Named tuple with fields 'tasks' and 'ch_avail'.
        solutions : Collection of SchedulingSolution
            Named tuple with fields 'sch', 'loss', and 't_run'.
        file_path : os.PathLike or str, optional
            File location relative to data/schedules/

        """

        save_dict = {
            'n_tasks': self.n_tasks,
            'n_ch': self.n_ch,
            'task_gen': self.task_gen,
            'ch_avail_gen': self.ch_avail_gen,
            'problems': problems
        }
        if solutions is not None:
            save_dict['solutions'] = solutions

        file_path = Path(file_path)

        try:  # search for existing file
            with file_path.open(mode='rb') as fid:
                load_dict = dill.load(fid)

            # Check equivalence of generators
            conditions = [
                load_dict['n_tasks'] == save_dict['n_tasks'],
                load_dict['n_ch'] == save_dict['n_ch'],
                load_dict['task_gen'] == save_dict['task_gen'],
                load_dict['ch_avail_gen'] == save_dict['ch_avail_gen']
            ]

            if all(conditions):  # Append loaded problems and solutions
                print('File already exists. Appending existing data.')

                save_dict['problems'] += load_dict['problems']

                if 'solutions' in save_dict.keys():
                    if 'solutions' in load_dict.keys():
                        save_dict['solutions'] += load_dict['solutions']
                    else:
                        save_dict['solutions'] += [None for __ in range(len(load_dict['problems']))]
                elif 'solutions' in load_dict.keys():
                    save_dict['solutions'] = [None for __ in range(len(save_dict['problems']))] + load_dict['solutions']

        except FileNotFoundError:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open(mode='wb') as fid:
            dill.dump(save_dict, fid)  # save schedules

    def __eq__(self, other):
        if isinstance(other, Base):
            conditions = [
                self.n_tasks == other.n_tasks,
                self.n_ch == other.n_ch,
                self.task_gen == other.task_gen,
                self.ch_avail_gen == other.ch_avail_gen
            ]
            return all(conditions)
        else:
            return NotImplemented

    def summary(self):
        cls_str = self.__class__.__name__

        plural_ = 's' if self.n_ch > 1 else ''
        str_ = f"{cls_str}\n---\n{self.n_ch} channel{plural_}, {self.n_tasks} tasks"

        if self.ch_avail_gen is not None:
            str_ += '\n\n' + self.ch_avail_gen.summary()
        if self.task_gen is not None:
            str_ += '\n\n' + self.task_gen.summary()

        return str_


class Random(Base):
    """Randomly generated scheduling problems."""

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        tasks = list(self.task_gen(self.n_tasks, rng=rng))
        ch_avail = list(self.ch_avail_gen(self.n_ch, rng=rng))

        return SchedulingProblem(tasks, ch_avail)

    @classmethod
    def _task_gen_factory(cls, n_tasks, task_gen, n_ch, ch_avail_lim, rng):
        ch_avail_gen = chan_gens.UniformIID(lims=ch_avail_lim)
        return cls(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

    @classmethod
    def continuous_linear(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **task_gen_kwargs):
        task_gen = task_gens.ContinuousUniformIID.linear(**task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    @classmethod
    def continuous_linear_drop(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **task_gen_kwargs):
        task_gen = task_gens.ContinuousUniformIID.linear_drop(**task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    @classmethod
    def discrete_linear(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **task_gen_kwargs):
        task_gen = task_gens.DiscreteIID.linear_uniform(**task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    @classmethod
    def discrete_linear_drop(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **task_gen_kwargs):
        task_gen = task_gens.DiscreteIID.linear_drop_uniform(**task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    # @classmethod
    # def search_track(cls, n_tasks, n_ch, p=None, t_release_lim=(0., .018), ch_avail_lim=(0., 0.), rng=None):
    #     task_gen = task_gens.SearchTrackIID(p, t_release_lim)
    #     return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)
    #
    # @classmethod
    # def radar(cls, n_tasks, n_ch, mode, ch_avail_lim=(0., 0.), rng=None):
    #     task_gen = task_gens.Radar(mode)
    #     return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)


class FixedTasks(Base, ABC):
    cls_task_gen = None

    def __init__(self, n_tasks, n_ch, task_gen, ch_avail_gen, rng=None):
        """
        Problem generators with fixed set of tasks.

        Parameters
        ----------
        n_tasks : int
            Number of tasks.
        n_ch: int
            Number of channels.
        task_gen : generators.tasks.Permutation
            Task generation object.
        ch_avail_gen : generators.channels.Deterministic
            Returns random initial channel availabilities.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        """

        super().__init__(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

        self._check_task_gen(task_gen)
        if not isinstance(ch_avail_gen, chan_gens.Deterministic):
            raise TypeError("Channel generator must be Deterministic.")

        self.problem = SchedulingProblem(task_gen.tasks, ch_avail_gen.ch_avail)
        self._solution = None

    @abstractmethod
    def _check_task_gen(self, task_gen):
        raise NotImplementedError

    @property
    def solution(self):
        """Solution for the fixed task set. Performs Branch-and-Bound the first time the property is accessed."""
        if self._solution is None:
            self._solution = super()._gen_solution(self.problem, verbose=True)
        return self._solution

    @abstractmethod
    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        raise NotImplementedError

    @classmethod
    def _task_gen_factory(cls, n_tasks, task_gen, n_ch, rng):
        ch_avail_gen = chan_gens.Deterministic.from_uniform(n_ch)
        return cls(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

    @classmethod
    def continuous_linear_drop(cls, n_tasks, n_ch, rng=None, **task_gen_kwargs):
        task_gen = cls.cls_task_gen.continuous_linear_drop(n_tasks, **task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, rng)

    @classmethod
    def discrete_linear_drop(cls, n_tasks, n_ch, rng=None, **task_gen_kwargs):
        task_gen = cls.cls_task_gen.discrete_linear_drop(n_tasks, **task_gen_kwargs)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, rng)

    @classmethod
    def search_track(cls, n_tasks, n_ch, probs=None, t_release_lim=(0., 0.), rng=None):
        task_gen = cls.cls_task_gen.search_track(n_tasks, probs, t_release_lim)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, rng)


class DeterministicTasks(FixedTasks):
    cls_task_gen = task_gens.Deterministic

    def _check_task_gen(self, task_gen):
        if not isinstance(task_gen, task_gens.Deterministic):
            raise TypeError

    def _gen_problem(self, rng):
        return self.problem

    def _gen_solution(self, problem, verbose=False):
        return self.solution


class PermutedTasks(FixedTasks):
    cls_task_gen = task_gens.Permutation

    def _check_task_gen(self, task_gen):
        if not isinstance(task_gen, task_gens.Permutation):
            raise TypeError

    def _gen_problem(self, rng):
        tasks = list(self.task_gen(self.n_tasks, rng=rng))
        return SchedulingProblem(tasks, self.problem.ch_avail)

    def _gen_solution(self, problem, verbose=False):
        idx = []  # permutation indices
        tasks_init = self.problem.tasks.copy()
        for task in problem.tasks:
            i = tasks_init.index(task)
            idx.append(i)
            tasks_init[i] = None  # ensures unique indices

        return SchedulingSolution(self.solution.sch[idx], self.solution.loss, self.solution.t_run)


class Dataset(Base):
    stack: deque[tuple]

    def __init__(self, problems, solutions=None, shuffle=False, repeat=False, task_gen=None, ch_avail_gen=None,
                 rng=None):

        n_tasks = len(problems[0].tasks)
        n_ch = len(problems[0].ch_avail)

        super().__init__(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

        self.stack = deque()
        self.add_problems(problems, solutions)

        if shuffle:
            self.shuffle()

        self.repeat = repeat

    n_problems = property(lambda self: len(self.stack))

    @classmethod
    def load(cls, file_path, shuffle=False, repeat=False, rng=None):
        """Load problems/solutions from memory."""

        with Path(file_path).open(mode='rb') as fid:
            dict_gen = dill.load(fid)

        args = [dict_gen['problems']]
        if 'solutions' in dict_gen.keys():
            args.append(dict_gen['solutions'])
        kwargs = {'shuffle': shuffle, 'repeat': repeat, 'task_gen': dict_gen['task_gen'],
                  'ch_avail_gen': dict_gen['ch_avail_gen'], 'rng': rng}
        return cls(*args, **kwargs)

    def pop_dataset(self, n, shuffle=False, repeat=False, rng=None):
        """Create a new Dataset from elements of own queue."""

        if isinstance(n, float):  # interpret as fraction of total problems
            n *= self.n_problems

        items = [self.stack.pop() for __ in range(n)]
        problems, solutions = zip(*items)
        return Dataset(problems, solutions, shuffle, repeat, self.task_gen, self.ch_avail_gen, rng)

    def add_problems(self, problems, solutions=None):
        """Add problems and solutions to the data set."""

        if solutions is None:
            solutions = [None for __ in range(len(problems))]
        elif len(solutions) != len(problems):
            raise ValueError("Number of solutions must equal the number of problems.")

        self.stack.extendleft(zip(problems, solutions))

    def shuffle(self, rng=None):
        """Shuffle problems and solutions in-place."""

        rng = self._get_rng(rng)
        _temp = rng.permutation(np.array(self.stack, dtype=object))
        self.stack = deque(map(tuple, _temp))

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        if self.n_problems == 0:
            raise ValueError("Problem generator data has been exhausted.")

        self._last_item = self.stack.pop()
        if self.repeat:
            self.stack.appendleft(self._last_item)

        return self._last_item[0]

    def _gen_solution(self, problem, verbose=False):
        _last_problem, _last_solution = self._last_item
        if problem != _last_problem:
            raise ValueError("Bug: problem used by `_gen_solution` should match dataset `_last_solution`.")

        if _last_solution is not None:
            return _last_solution
        else:  # use B&B solver
            solution = super()._gen_solution(problem, verbose)
            if self.repeat:  # store result
                self.stack[0] = (_last_problem, solution)  # at index 0 after `appendleft` in `_gen_problem`
            return solution

    def summary(self):
        str_ = super().summary()
        str_ += f"\n\n- Number of problems: {self.n_problems}"
        return str_

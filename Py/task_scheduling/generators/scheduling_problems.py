"""Generator objects for complete tasking problems with optimal solutions."""

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque
from time import strftime
from typing import Iterable
from functools import partial
import dill
from pathlib import Path

import numpy as np
import pandas as pd

import task_scheduling.tasks
from task_scheduling.algorithms.free import branch_bound, earliest_release
from task_scheduling.util.generic import RandomGeneratorMixin, timing_wrapper, SchedulingProblem, SchedulingSolution
from task_scheduling.generators import tasks as task_gens, channel_availabilities as chan_gens

np.set_printoptions(precision=2)
pd.options.display.float_format = '{:,.2f}'.format      # TODO: global??

data_path = Path.cwd() / 'data' / 'schedules'


class Base(RandomGeneratorMixin, ABC):
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
        ch_avail_gen : generators.channel_availabilities.Base
            Returns random initial channel availabilities.
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        """

        super().__init__(rng)

        self.n_tasks = n_tasks
        self.n_ch = n_ch
        self.task_gen = task_gen
        self.ch_avail_gen = ch_avail_gen

    def __call__(self, n_gen, solve=False, verbose=False, save=False, file=None, rng=None):
        """
        Call problem generator.

        Parameters
        ----------
        n_gen : int
            Number of scheduling problems to generate.
        solve : bool, optional
            Enables generation of Branch & Bound optimal solutions.
        verbose : bool, optional
            Enables print-out of optimal scheduler progress.
        save : bool, optional
            Enables serialization of generated problems/solutions.
        file : str, optional
            File location relative to data/schedules/
        rng : int or RandomState or Generator, optional
            NumPy random number generator or seed. Instance RNG if None.

        Yields
        ------
        SchedulingProblem
            Scheduling problem namedtuple.
        SchedulingSolution, optional
            Scheduling solution namedtuple.

        """

        save_dict = {'problems': [], 'solutions': [] if solve else None,
                     'task_gen': self.task_gen, 'ch_avail_gen': self.ch_avail_gen}

        # Generate tasks and find optimal schedules
        rng = self._get_rng(rng)
        for i_gen in range(n_gen):
            if verbose:
                print(f'Scheduling Problem: {i_gen + 1}/{n_gen}', end='\n')

            problem = self._gen_problem(rng)
            if save:
                save_dict['problems'].append(problem)

            if solve:
                solution = self._gen_solution(problem, verbose)
                if save:
                    save_dict['solutions'].append(solution)

                yield problem, solution
            else:
                yield problem

        if save:
            self.save(save_dict, file)

    @abstractmethod
    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        raise NotImplementedError

    @staticmethod
    def _gen_solution(problem, verbose=False):
        t_ex, ch_ex, t_run = timing_wrapper(partial(branch_bound, verbose=verbose))(*problem)
        return SchedulingSolution(t_ex, ch_ex, t_run)

    @staticmethod
    def save(save_dict, file=None):
        """
        Serialize scheduling problems/solutions.

        Parameters
        ----------
        save_dict: dict
            Serialized dict with keys 'problems', 'solutions', 'task_gen', and 'ch_avail_gen'.
        file : str, optional
            File location relative to data/schedules/

        """

        if file is None:
            file = f"temp/{strftime('%Y-%m-%d_%H-%M-%S')}"
        file_path = data_path.joinpath(file)

        try:    # search for existing file
            with file_path.open(mode='rb') as fid:
                load_dict = dill.load(fid)

            # Check equivalence of generators
            conditions = [load_dict['task_gen'] == save_dict['task_gen'],
                          load_dict['ch_avail_gen'] == save_dict['ch_avail_gen'],
                          len(load_dict['problems'][0].tasks) == len(save_dict['problems'][0].tasks),
                          len(load_dict['problems'][0].ch_avail) == len(save_dict['problems'][0].ch_avail),
                          ]

            if all(conditions):     # Append loaded problems and solutions
                print('File already exists. Appending existing data.')

                if 'solutions' in save_dict.keys():     # FIXME: check!!
                    try:
                        save_dict['solutions'] += load_dict['solutions']
                        save_dict['problems'] += load_dict['problems']
                    except KeyError:
                        pass    # Skip if new data has solutions and loaded data does not
                else:
                    save_dict['problems'] += load_dict['problems']

        except FileNotFoundError:
            file_path.parent.mkdir(exist_ok=True)

        with data_path.joinpath(file).open(mode='wb') as fid:
            dill.dump(save_dict, fid)  # save schedules

    def __eq__(self, other):
        if isinstance(other, Base):
            conditions = [self.n_tasks == other.n_tasks,
                          self.n_ch == other.n_ch,
                          self.task_gen == other.task_gen,
                          self.ch_avail_gen == other.ch_avail_gen]
            return all(conditions)
        else:
            return NotImplemented


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
    def continuous_relu_drop(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **relu_lims):
        task_gen = task_gens.ContinuousUniformIID.relu_drop(**relu_lims)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    @classmethod
    def discrete_relu_drop(cls, n_tasks, n_ch, ch_avail_lim=(0., 0.), rng=None, **relu_vals):
        task_gen = task_gens.DiscreteIID.relu_drop_uniform(**relu_vals)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)

    @classmethod
    def search_track(cls, n_tasks, n_ch, probs=None, t_release_lim=(0., .018), ch_avail_lim=(0., 0.), rng=None):
        task_gen = task_gens.SearchTrackIID(probs, t_release_lim)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, ch_avail_lim, rng)


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
        ch_avail_gen : generators.channel_availabilities.Deterministic
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
    def continuous_relu_drop(cls, n_tasks, n_ch, rng=None, **relu_lims):
        task_gen = cls.cls_task_gen.continuous_relu_drop(n_tasks, **relu_lims)
        return cls._task_gen_factory(n_tasks, task_gen, n_ch, rng)

    @classmethod
    def discrete_relu_drop(cls, n_tasks, n_ch, rng=None, **relu_vals):
        task_gen = cls.cls_task_gen.discrete_relu_drop(n_tasks, **relu_vals)
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

        return SchedulingSolution(self.solution.t_ex[idx], self.solution.ch_ex[idx], self.solution.t_run)


class Dataset(Base):
    def __init__(self, problems, solutions=None, shuffle=False, repeat=False, task_gen=None, ch_avail_gen=None,
                 rng=None):

        n_tasks, n_ch = len(problems[0].tasks), len(problems[0].ch_avail)
        super().__init__(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

        # self.problems = deque(problems)
        # self.solutions = deque(solutions) if solutions is not None else None
        self.problems = deque()     # TODO: single deque?
        self.solutions = deque()
        self.add_problems(problems, solutions)

        if shuffle:
            self.shuffle()

        self.repeat = repeat

    n_problems = property(lambda self: len(self.problems))

    def add_problems(self, problems, solutions=None):
        self.problems.extendleft(problems)

        if solutions is None:
            solutions = [None for __ in range(len(problems))]
        self.solutions.extendleft(solutions)

    @classmethod
    def load(cls, file, shuffle=False, repeat=False, rng=None):
        """Load problems/solutions from memory."""

        # TODO: loads entire data set into memory - need iterative read/yield for large data sets
        with data_path.joinpath(file).open(mode='rb') as fid:
            dict_gen = dill.load(fid)

        return cls(**dict_gen, shuffle=shuffle, repeat=repeat, rng=rng)

    def shuffle(self, rng=None):
        rng = self._get_rng(rng)

        _temp = np.array(list(zip(self.problems, self.solutions)), dtype=np.object)
        _p, _s = zip(*rng.permutation(_temp).tolist())
        self.problems, self.solutions = deque(_p), deque(_s)

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        if len(self.problems) == 0:
            raise ValueError("Problem generator data has been exhausted.")

        problem = self.problems.pop()
        self._solution_i = self.solutions.pop()

        if self.repeat:
            self.problems.appendleft(problem)
            self.solutions.appendleft(self._solution_i)

        return problem

    def _gen_solution(self, problem, verbose=False):
        if self._solution_i is not None:
            return self._solution_i
        else:   # use B&B solver
            solution = super()._gen_solution(problem, verbose)
            if self.repeat:     # store result
                self.solutions[0] = solution        # at index 0 after `appendleft` in `_gen_problem`
            return solution







class QueueFlexDAR(Base):
    def __init__(self, n_tasks, tasks_full, ch_avail, RP=0.04, clock=0, scheduler=earliest_release, record_revisit=True):

        self._cls_task = task_scheduling.tasks.check_task_types(tasks_full)

        # FIXME: make a task_gen???
        super().__init__(n_tasks, len(ch_avail), task_gen=None, ch_avail_gen=None, rng=None)

        self.queue = deque()
        self.add_tasks(tasks_full)
        self.ch_avail = np.array(ch_avail, dtype=np.float)
        self.clock = np.array(0, dtype=np.float)
        self.RP = RP
        self.record_revisit = record_revisit
        self.scheduler = scheduler

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""

        ch_avail_input = deepcopy(self.ch_avail)  # This is what you want to pass out in the scheduling problem
        self.reprioritize()  # Reprioritize
        tasks = [self.queue.pop() for _ in range(self.n_tasks)]  # Pop tasks

        t_ex, ch_ex, t_run = timing_wrapper(self.scheduler)(tasks, self.ch_avail)  # Scheduling using ERT

        # TODO: use t_run to check validity of t_ex
        # t_ex = np.max([t_ex, [t_run for _ in range(len(t_ex))]], axis=0)

        # obs, reward, done, info = env.step()

        # done = False
        # while not done:
        #     obs, reward, done, info = env.step(action)

        self.updateFlexDAR(deepcopy(tasks), t_ex, ch_ex)  # Add tasks back on queue
        self.clock += self.RP  # Update clock

        # TODO: add prioritization?

        return SchedulingProblem(tasks, ch_avail_input.copy())  # SchedulingProblemFlexDAR(tasks, self.ch_avail.copy(), self.clock.copy())

    def add_tasks(self, tasks):
        if isinstance(tasks, Iterable):
            self.queue.extendleft(tasks)
        else:
            self.queue.appendleft(tasks)    # for single tasks

    def update(self, tasks, t_ex, ch_ex):
        for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
            task.t_release = t_ex_i + task.duration
            self.ch_avail[int(ch_ex_i)] = max(self.ch_avail[int(ch_ex_i)], task.t_release)
            self.add_tasks(task)

        # for task, t_ex_i in zip(tasks, t_ex):
        #     task.t_release = t_ex_i + task.duration
        #
        # for ch in range(self.n_ch):
        #     tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        #     self.ch_avail[ch] = max(self.ch_avail[ch], *(task.t_release for task in tasks_ch))
        #
        # self.add_tasks(tasks)

    def updateFlexDAR(self, tasks, t_ex, ch_ex):
        for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
            # duration = np.array([task.duration for task in job_scheduler])
            # executed_tasks = t_complete <= timeSec + RP # Task that are executed
            t_complete_i = t_ex_i + task.duration
            if t_complete_i <= self.RP + self.clock:
                task.t_release = t_ex_i + task.duration
                if self.record_revisit:
                    task.revisit_times.append(t_ex_i)
                # task.count_revisit += 1  Node need as count is = len(revisit_times) in ReluDropRadar
                self.ch_avail[ch_ex_i] = max(self.ch_avail[ch_ex_i], task.t_release)
                self.add_tasks(task)
            else:
                self.add_tasks(task)

        # self.clock += self.RP # Update Overall Clock

        # for task, t_ex_i in zip(tasks, t_ex):
        #     task.t_release = t_ex_i + task.duration
        #
        # for ch in range(self.n_ch):
        #     tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        #     self.ch_avail[ch] = max(self.ch_avail[ch], *(task.t_release for task in tasks_ch))
        #
        # self.add_tasks(tasks)

    def reprioritize(self):

        # Evaluate tasks at current time
        # clock = 1 # For debugging
        priority = np.array([task(self.clock) for task in self.queue])
        index = np.argsort(-1*priority, kind='mergesort')  # -1 used to reverse order
        tasks = []
        tasks_sorted = []
        for task in self.queue:
            tasks.append(task)

        tasks_sorted = [self.queue[idx] for idx in index]

        # for idx in range(len(self.queue)):
        #     task = self.queue[index[idx]]
        #     tasks_sorted = tasks_sorted.append(task)

        self.queue.clear()
        self.add_tasks(tasks_sorted)

    def summary(self):
        print(f"Channel availabilities: {self.ch_avail}")
        print(f"Task queue:")
        df = pd.DataFrame({name: [getattr(task, name) for task in self.queue]
                           for name in self._cls_task.param_names})
        priority = np.array([task(self.clock) for task in self.queue])
        df['priority'] = priority
        print(df)

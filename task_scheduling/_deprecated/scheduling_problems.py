# from task_scheduling.generators.scheduling_problems import Base


class DatasetOld(Base):  # TODO: deprecate?
    """
    Generator of scheduling problems in memory.

    Parameters
    ----------
    problems : List of SchedulingProblem
        Scheduling problems
    solutions : List of SchedulingSolution, optional
        Optimal scheduling solutions
    task_gen : generators.tasks.BaseIID, optional
        Task generation object.
    ch_avail_gen : generators.channel_availabilities.BaseIID, optional
        Returns random initial channel availabilities.
    iter_mode : str, optional
        If 'once', generator call raises a warning when all data has been yielded. If 'repeat', a new pass is started.
    shuffle_mode : str, optional
        If 'once', data is randomly permuted during initialization. If 'repeat', data is permuted every complete pass.
    rng : int or RandomState or Generator, optional
        Random number generator seed or object.

    """

    def __init__(
        self,
        problems,
        solutions=None,
        task_gen=None,
        ch_avail_gen=None,
        iter_mode="once",
        shuffle_mode="never",
        rng=None,
    ):

        self.problems = problems
        self.solutions = solutions

        n_tasks, n_ch = len(problems[0].tasks), len(problems[0].ch_avail)
        super().__init__(n_tasks, n_ch, task_gen, ch_avail_gen, rng)

        self.iter_mode = iter_mode
        self.shuffle_mode = shuffle_mode

        self.i = None
        self.n_problems = len(self.problems)
        self.restart(self.shuffle_mode in ("once", "repeat"))

    @classmethod
    def load(cls, file, iter_mode="once", shuffle_mode="never", rng=None):
        """Load problems/solutions from memory."""

        # TODO: loads entire data set into memory - need iterative read/yield for large data sets
        with data_path.joinpath(file).open(mode="rb") as fid:
            dict_gen = dill.load(fid)

        return cls(**dict_gen, iter_mode=iter_mode, shuffle_mode=shuffle_mode, rng=rng)

    def restart(self, shuffle=False, rng=None):
        """Resets data index pointer to beginning, performs optional data shuffle."""
        self.i = 0
        if shuffle:
            rng = self._get_rng(rng)
            if self.solutions is None:
                self.problems = rng.permutation(self.problems).tolist()
            else:
                # _temp = list(zip(self.problems, self.solutions))
                _temp = np.array(list(zip(self.problems, self.solutions)), dtype=np.object)
                _p, _s = zip(*rng.permutation(_temp).tolist())
                self.problems, self.solutions = list(_p), list(_s)

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        if self.i == self.n_problems:
            if self.iter_mode == "once":
                raise ValueError("Problem generator data has been exhausted.")
            elif self.iter_mode == "repeat":
                self.restart(self.shuffle_mode == "repeat", rng=rng)

        problem = self.problems[self.i]
        if self.solutions is not None:
            self._solution_i = self.solutions[self.i]
        else:
            self._solution_i = None

        self.i += 1
        return problem

    def _gen_solution(self, problem, verbose=False):
        if self._solution_i is not None:
            return self._solution_i
        else:
            return super()._gen_solution(problem, verbose)


class Queue(Base):  # TODO: deprecate in favor of generators.tasks.Dataset?
    def __init__(self, n_tasks, tasks_full, ch_avail):

        self._cls_task = task_gens.check_task_types(tasks_full)

        # FIXME: make a task_gen???
        super().__init__(n_tasks, len(ch_avail), task_gen=None, ch_avail_gen=None, rng=None)

        self.queue = deque()
        self.add_tasks(tasks_full)
        self.ch_avail = ch_avail

    def _gen_problem(self, rng):
        """Return a single scheduling problem (and optional solution)."""
        tasks = [self.queue.pop() for _ in range(self.n_tasks)]

        return SchedulingProblem(tasks, self.ch_avail)

    def add_tasks(self, tasks):
        if isinstance(tasks, Iterable):
            self.queue.extendleft(tasks)
        else:
            self.queue.appendleft(tasks)  # for single tasks

    def update(self, tasks, t_ex, ch_ex):
        for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
            task.t_release = t_ex_i + task.duration
            self.ch_avail[ch_ex_i] = max(self.ch_avail[ch_ex_i], task.t_release)
            self.add_tasks(task)

        # for task, t_ex_i in zip(tasks, t_ex):
        #     task.t_release = t_ex_i + task.duration
        #
        # for ch in range(self.n_ch):
        #     tasks_ch = np.array(tasks)[ch_ex == ch].tolist()
        #     self.ch_avail[ch] = max(self.ch_avail[ch], *(task.t_release for task in tasks_ch))
        #
        # self.add_tasks(tasks)

    def summary(self):
        print(f"Channel availabilities: {self.ch_avail}")
        print(f"Task queue:")
        df = pd.DataFrame(
            {
                name: [getattr(task, name) for task in self.queue]
                for name in self._cls_task.param_names
            }
        )
        print(df)


# class Dataset(Base):
#     def __init__(self, problems, solutions=None, shuffle=False, repeat=False, task_gen=None, ch_avail_gen=None,
#                  rng=None):
#
#         n_tasks = len(problems[0].tasks)
#         n_ch = len(problems[0].ch_avail)
#
#         super().__init__(n_tasks, n_ch, task_gen, ch_avail_gen, rng)
#
#         self.problems = deque()  # TODO: single deque?
#         self.solutions = deque()
#         self.add_problems(problems, solutions)
#
#         if shuffle:
#             self.shuffle()
#
#         self.repeat = repeat
#
#     n_problems = property(lambda self: len(self.problems))
#
#     @classmethod
#     def load(cls, file_path, shuffle=False, repeat=False, rng=None):
#         """Load problems/solutions from memory."""
#
#         with Path(file_path).open(mode='rb') as fid:
#             dict_gen = dill.load(fid)
#
#         args = [dict_gen['problems']]
#         if 'solutions' in dict_gen.keys():
#             args.append(dict_gen['solutions'])
#         kwargs = {'shuffle': shuffle, 'repeat': repeat, 'task_gen': dict_gen['task_gen'],
#                   'ch_avail_gen': dict_gen['ch_avail_gen'], 'rng': rng}
#         return cls(*args, **kwargs)
#
#     def pop_dataset(self, n, shuffle=False, repeat=False, rng=None):
#         """Create a new Dataset from elements of own queue."""
#
#         if isinstance(n, float):  # interpret as fraction of total problems
#             n *= self.n_problems
#
#         problems = [self.problems.pop() for __ in range(n)]
#         solutions = [self.solutions.pop() for __ in range(n)]
#         return Dataset(problems, solutions, shuffle, repeat, self.task_gen, self.ch_avail_gen, rng)
#
#     def add_problems(self, problems, solutions=None):
#         """Add problems and solutions to the data set."""
#
#         self.problems.extendleft(problems)
#
#         if solutions is None:
#             solutions = [None for __ in range(len(problems))]
#         elif len(solutions) != len(problems):
#             raise ValueError("Number of solutions must equal the number of problems.")
#
#         self.solutions.extendleft(solutions)
#
#     def shuffle(self, rng=None):
#         """Shuffle problems and solutions in-place."""
#
#         rng = self._get_rng(rng)
#
#         _temp = np.array(list(zip(self.problems, self.solutions)), dtype=object)
#         _p, _s = zip(*rng.permutation(_temp).tolist())
#         self.problems, self.solutions = deque(_p), deque(_s)
#
#     def _gen_problem(self, rng):
#         """Return a single scheduling problem (and optional solution)."""
#         if self.n_problems == 0:
#             raise ValueError("Problem generator data has been exhausted.")
#
#         problem = self.problems.pop()
#         self._solution_i = self.solutions.pop()
#
#         if self.repeat:
#             self.problems.appendleft(problem)
#             self.solutions.appendleft(self._solution_i)
#
#         return problem
#
#     def _gen_solution(self, problem, verbose=False):
#         if self._solution_i is not None:
#             return self._solution_i
#         else:  # use B&B solver
#             solution = super()._gen_solution(problem, verbose)
#             if self.repeat:  # store result
#                 self.solutions[0] = solution  # at index 0 after `appendleft` in `_gen_problem`
#             return solution
#
#     def summary(self, file=None):
#         super().summary(file)
#         print(f"Number of problems: {self.n_problems}\n", file=file)

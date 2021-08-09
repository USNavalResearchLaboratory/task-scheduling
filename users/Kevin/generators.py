from collections import deque
from copy import deepcopy
from typing import Iterable

import numpy as np
import pandas as pd
from gym import spaces


#%% Task generators
class FlexDAR(Base):
    def __init__(self, n_track=0, param_spaces=None, rng=None):
        super().__init__(cls_task=task_types.ReluDropRadar, param_spaces=None, rng=None)

        self.n_track = n_track
        tasks_full = []

        # Search tasks
        n_beams_per_row = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
        t_dwells = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18]) * 1e-3
        dwell_types = ['HS', *('AHS' for _ in range(8))]
        for n_beams, t_dwell, dwell_type in zip(n_beams_per_row, t_dwells, dwell_types):
            tasks_full.extend([self.cls_task.search(t_dwell, dwell_type) for _ in range(n_beams)])

        # Track tasks
        for slant_range, range_rate in zip(self.rng.uniform(0, 200, n_track), self.rng.uniform(-343, 343, n_track)):
            # tasks_full.append(self.cls_task.track_from_kinematics(slant_range, range_rate))
            # Current classmethod called "from_kinematics"
            tasks_full.append(self.cls_task.from_kinematics(slant_range, range_rate))

        self.tasks_full = tasks_full

        # # Generate Search Tasks
        # SearchParams = TaskParameters()
        # SearchParams.NbeamsPerRow = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
        # # SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload
        # SearchParams.DwellTime = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18]) * 1e-3
        # SearchParams.RevistRate = np.array([2.5, 5, 5, 5, 5, 5, 5, 5, 5])
        # SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1  # Upper Bound on Revisit Rate
        # SearchParams.Penalty = 300 * np.ones(np.shape(SearchParams.RevistRate))  # Penalty for exceeding UB
        # SearchParams.Slope = 1. / SearchParams.RevistRate
        #
        # n_search = np.sum(SearchParams.NbeamsPerRow)
        # SearchParams.JobDuration = np.array([])
        # SearchParams.JobSlope = np.array([])
        # SearchParams.DropTime = np.array([])  # Task dropping time. Will get updated as tasks get processed
        # # Used to update DropTimes. Fixed for a given task e.x. always 2.6 process task at time 1 DropTime becomes 3.6
        # # SearchParams.DropTimeFixed = np.array([])
        # SearchParams.DropCost = np.array([])
        # for jj in range(len(SearchParams.NbeamsPerRow)):
        #     SearchParams.JobDuration = np.append(SearchParams.JobDuration,
        #                                          np.repeat(SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
        #     SearchParams.JobSlope = np.append(SearchParams.JobSlope,
        #                                       np.repeat(SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
        #     SearchParams.DropTime = np.append(SearchParams.DropTime,
        #                                       np.repeat(SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
        #     # SearchParams.DropTimeFixed = np.append(SearchParams.DropTimeFixed, np.repeat(SearchParams.RevisitRateUB[jj],
        #     #                                                                              SearchParams.NbeamsPerRow[jj]))
        #     SearchParams.DropCost = np.append(SearchParams.DropCost,
        #                                       np.repeat(SearchParams.Penalty[jj], SearchParams.NbeamsPerRow[jj]))

        # # Generate Track Tasks
        # TrackParams = TaskParameters()  # Initializes to something like matlab structure
        # # Ntrack = 10
        #
        # # Spawn tracks with uniformly distributed ranges and velocity
        # MaxRangeNmi = 200  #
        # MaxRangeRateMps = 343  # Mach 1 in Mps is 343
        #
        # truth = TaskParameters
        # truth.rangeNmi = MaxRangeNmi * self.rng.uniform(0, 1, n_track)
        # truth.rangeRateMps = 2 * MaxRangeRateMps * self.rng.uniform(0, 1, n_track) - MaxRangeRateMps
        #
        # TrackParams.DwellTime = np.array([18, 18, 18]) * 1e-3
        # TrackParams.RevisitRate = np.array([1, 2, 4])
        # TrackParams.RevisitRateUB = TrackParams.RevisitRate + 0.1
        # TrackParams.Penalty = 300 * np.ones(np.shape(TrackParams.DwellTime))
        # TrackParams.Slope = 1. / TrackParams.RevisitRate
        # TrackParams.JobDuration = []
        # TrackParams.JobSlope = []
        # TrackParams.DropTime = []
        # TrackParams.DropTimeFixed = []
        # TrackParams.DropCost = []
        # for jj in range(n_track):
        #     if truth.rangeNmi[jj] <= 50:
        #         TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[0])
        #         TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[0])
        #         TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[0])
        #         TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[0])
        #         TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[0])
        #     elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
        #         TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[1])
        #         TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
        #         TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
        #         TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[1])
        #         TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
        #     else:
        #         TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[2])
        #         TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
        #         TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
        #         TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[2])
        #         TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])

    def __call__(self, rng=None):

        n_track = self.n_track

        # Begin Scheduler Loop

        # rng = self.rng.default_rng(100)
        # task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
        #                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
        # tasks = task_gen.rand_tasks(N)

        # A = list()
        tasks = []
        cnt = 0  # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
        for ii in range(Nsearch):
            # job.append(0, self.cls_task(SearchParams.JobDuration[ii], SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
            tasks.append(
                self.cls_task(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii],
                              SearchParams.DropCost[ii]))
            tasks[ii].Id = cnt  # Numeric Identifier for each job
            cnt = cnt + 1
            if tasks[ii].slope == 0.4:
                tasks[ii].Type = 'HS'  # Horizon Search (Used to determine revisit rates by job type
            else:
                tasks[ii].Type = 'AHS'  # Above horizon search
            tasks[ii].Priority = tasks[ii](0)  # Priority used to select which jobs to give to scheduler

            # tasks = self.cls_task(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
            # A.append(tasks)
            # del tasks

        for ii in range(n_track):
            # job.append(self.cls_task(0, TrackParams.JobDuration[ii], TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
            tasks.append(
                self.cls_task(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii],
                              TrackParams.DropCost[ii]))
            tasks[cnt].Id = cnt  # Numeric Identifier for each job
            if tasks[cnt].slope == 0.25:
                tasks[cnt].Type = 'Tlow'  # Low Priority Track
            elif tasks[cnt].slope == 0.5:
                tasks[cnt].Type = 'Tmed'  # Medium Priority Track
            else:
                tasks[cnt].Type = 'Thigh'  # High Priority Track
            tasks[cnt].Priority = tasks[cnt](0)
            cnt = cnt + 1

            self.tasks = list(tasks)

            cls_task = self.tasks[0].__class__
            if not all(isinstance(task, cls_task) for task in self.tasks[1:]):
                raise TypeError("All tasks must be of the same type.")

            if param_lims is None:
                param_lims = {}
                for name in cls_task.param_names:
                    values = [getattr(task, name) for task in tasks]
                    param_lims[name] = (min(values), max(values))

            # super().__init__(cls_task, param_lims, rng)

            # self.tasks = tasks

        return tasks

    def summary(self):  # TODO: Fix this

        df = pd.DataFrame({name: [getattr(task, name) for task in self.tasks]
                           for name in self._cls_task.param_names})
        print(df)

    # for task in
    #     yield task


class FlexDARlike(Base):
    def __init__(self, n_track=0, param_spaces=None, rng=None):

        # self.targets = [{'duration': .040, 't_revisit': 2.5},   # horizon search
        #                 {'duration': .040, 't_revisit': 5.0},   # above horizon search
        #                 {'duration': .030, 't_revisit': 5.0},   # above horizon search
        #                 {'duration': .020, 't_revisit': 5.0},  # above horizon search
        #                 {'duration': .010, 't_revisit': 5.0},  # above horizon search
        #                 {'duration': .020, 't_revisit': 1.0},   # high priority track
        #                 {'duration': .020, 't_revisit': 2.0},   # med priority track
        #                 {'duration': .020, 't_revisit': 4.0},   # low priority track
        #                 ]
        # t_release_lim = 50
        # durations, t_revisits = map(np.array, zip(*[target.values() for target in self.targets]))
        # Currently use continuous spaces, need to change
        param_spaces = {'duration': spaces.Box(0, 5, shape=(), dtype=float),
                        't_release': spaces.Box(0, 50, shape=(), dtype=float),
                        'slope': spaces.Box(0, 1, shape=(), dtype=float),
                        't_drop': spaces.Box(0, 6, shape=(), dtype=float),
                        'l_drop': DiscreteSet([300.])
                        }

        super().__init__(cls_task=task_types.ReluDropRadar, param_spaces=param_spaces, rng=None)

        self.n_track = n_track
        tasks_full = []

        # Search tasks
        n_beams_per_row = np.array([30, 20, 20, 10, 10, 10, 10, 10])
        t_dwells = np.array([40, 40, 30, 20, 20, 20, 20, 10]) * 1e-3
        dwell_types = ['HS', *('AHS' for _ in range(7))]
        for n_beams, t_dwell, dwell_type in zip(n_beams_per_row, t_dwells, dwell_types):
            tasks_full.extend([self.cls_task.search(t_dwell, dwell_type) for _ in range(n_beams)])

        # Track tasks
        for slant_range, range_rate in zip(self.rng.uniform(0, 200, n_track), self.rng.uniform(-343, 343, n_track)):
            # tasks_full.append(self.cls_task.track_from_kinematics(slant_range, range_rate))
            # Current classmethod called "from_kinematics"
            tasks_full.append(self.cls_task.from_kinematics_notional(slant_range, range_rate))

        self.tasks_full = tasks_full

    def __call__(self, rng=None):

        n_track = self.n_track
        tasks = []
        return tasks

        def summary(self):  # TODO: Fix this

            df = pd.DataFrame({name: [getattr(task, name) for task in self.tasks]
                               for name in self._cls_task.param_names})
            print(df)


# Scheduling problem generators
class QueueFlexDAR(Base):
    def __init__(self, n_tasks, tasks_full, ch_avail, RP=0.04, clock=0, scheduler=earliest_release,
                 record_revisit=True):

        self._cls_task = task_scheduling.tasks.check_task_types(tasks_full)

        # FIXME: make a task_gen???
        super().__init__(n_tasks, len(ch_avail), task_gen=None, ch_avail_gen=None, rng=None)

        self.queue = deque()
        self.add_tasks(tasks_full)
        self.ch_avail = np.array(ch_avail, dtype=float)
        self.clock = np.array(0, dtype=float)
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

        return SchedulingProblem(tasks, ch_avail_input.copy())

    def add_tasks(self, tasks):
        if isinstance(tasks, Iterable):
            self.queue.extendleft(tasks)
        else:
            self.queue.appendleft(tasks)  # for single tasks

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
        index = np.argsort(-1 * priority, kind='mergesort')  # -1 used to reverse order
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
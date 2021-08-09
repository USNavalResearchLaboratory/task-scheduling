import numpy as np
import matplotlib.pyplot as plt
import time     # TODO: use builtin module timeit instead? or cProfile?
import random
import math
# from SL_policy_Discrete import load_policy, wrap_policy
from tree_search import branch_bound, random_sequencer, earliest_release
from functools import partial
# from util.generic import algorithm_repr
from util.plot import plot_task_losses, plot_schedule, plot_loss_runtime, scatter_loss_runtime
from util.results import check_valid, eval_loss
from more_itertools import locate
from math import factorial, floor
# import numpy as np
from task_scheduling.tasks import ReluDrop
import gym
from gym.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete


##
class TaskParameters:  # Initializes to something like matlab structure. Enables dot indexing
    pass


class PriorityQueueDiscrete(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config=None):

        """
        N_track - number of tracks
        N_search is fixed to 120

        """
        if env_config:
            self.env_config = env_config
        else:
            alg_func = partial(earliest_release, do_swap=True)

            env_config = {"RP": 0.040,  # Resource Period
                          "MaxTime": 20,  # MaxTime to run an episode
                          "Ntrack": 40,  # Number of dedicated tracks
                          "N": 8,  # Number of jobs to process at any time
                          "K": 1,   # Number of Channels
                          "Nbins": 10, # Number of bins to discretize time variables
                          "scheduler": alg_func  # Function used to perform scheduling --> in the future use B&B or NN
                          }
            NumSteps = np.int(np.round(env_config.get("MaxTime") / env_config.get("RP")))
            ChannelAvailableTime = np.zeros(env_config.get("K"))
            env_config['NumSteps'] = NumSteps
            env_config['ChannelAvailableTime'] = ChannelAvailableTime

            self.env_config = env_config



        state = self.reset()
        self.state = state
        self.iter = 0

        N = self.env_config.get("N")
        M = self.M
        actions = []
        for jj in range(N):
            actions.append(M-jj)

        self.action_space = gym.spaces.MultiDiscrete(actions)  # Works like this, first action pull off one job out of M, next action pull off one job out of (M-1) remaining, ..., repeat until N jobs have been pulled

        # self.action_space = Tuple((Discrete(2), Box(-10, 10, (2,))))
        # self.observation_space = Box(-10, 10, (2, 2))

        Nbins = self.env_config.get('Nbins')
        array_duration = 2*np.ones(M)
        array_t_drop = 5*np.ones(M)
        array_slope = 5*np.ones(M)
        array_t_release = Nbins*np.ones(M)
        array = np.hstack((array_t_release, array_duration, array_t_drop, array_slope))

        self.observation_space = gym.spaces.MultiDiscrete(array)
        # self.observation_space = gym.spaces.Tuple([Box(low=-12, high=1.0, shape=(M,)), MultiDiscrete(array_duration), MultiDiscrete(array_t_drop), MultiDiscrete(array_slope)])
        # self.observation_space = gym.spaces.Tuple([Box(-12, 0, (1,)), Discrete(2), Discrete(5), Discrete(4), (-10, 10, (2, 2))])

        ## Metrics
        Job_Revisit_Count = np.zeros((len(self.job), 1))
        Job_Revisit_Time = []
        for ii in range(len(self.job)):  # Create a list of empty lists --> make it multi-dimensional to support different algorithms
            elem = []
            for jj in range(1):
                elem.append([])
            Job_Revisit_Time.append(elem)

        info = {"Job_Revisit_Count": Job_Revisit_Count,
                "Job_Revisit_Time": Job_Revisit_Time
                }

        self.info = info



    def map_features_to_state(self, features):

        M = self.M
        NUM_FEATS = self.NUM_FEATS

        # Map features to observation space
        unique_duration = np.unique(features[:, 1])
        unique_t_drop = np.unique(features[:, 2])
        unique_slope = np.unique(features[:, 4])
        low = -12
        high = 1
        Nbins = self.env_config.get('Nbins')
        increment = (high-low)/Nbins
        unique_t_release = np.empty(Nbins+1)
        for jj in range(Nbins+1):
            unique_t_release[jj] = low + increment*jj

        temp = np.empty((M, NUM_FEATS - 1))
        # temp[:, 0] = features[:, 0]
        for jj in range(Nbins):
            idx = np.where((features[:, 0] > unique_t_release[jj]) & (features[:, 0] < unique_t_release[jj+1]) )
            temp[idx, 0] = jj

        for jj in range(len(unique_duration)):
            idx = np.where(features[:, 1] == unique_duration[jj])
            temp[idx, 1] = jj

        for jj in range(len(unique_t_drop)):
            idx = np.where(features[:, 2] == unique_t_drop[jj])
            temp[idx, 2] = jj

        for jj in range(len(unique_slope)):
            idx = np.where(features[:, 4] == unique_slope[jj])
            temp[idx, 3] = jj

        # state = (np.array(temp[:, 0], dtype='f'), np.array(temp[:, 1], dtype='int64'),
        #          np.array(temp[:, 2], dtype='int64'),
        #          np.array(temp[:, 3], dtype='int64'))

        state = np.hstack((np.array(temp[:, 0], dtype='int64'), np.array(temp[:, 1], dtype='int64'),
                np.array(temp[:, 2], dtype='int64'),
                np.array(temp[:, 3], dtype='int64')))

        return state


    def reset(self):

        RP = self.env_config.get("RP")
        Ntrack = self.env_config.get("Ntrack")
        MaxTime = self.env_config.get("MaxTime")

        random.seed(30)


        ## Generate Search Tasks
        SearchParams = TaskParameters()
        SearchParams.NbeamsPerRow = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
        # SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload
        SearchParams.DwellTime = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18]) * 1e-3
        SearchParams.RevistRate = np.array([2.5, 5, 5, 5, 5, 5, 5, 5, 5])
        SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1  # Upper Bound on Revisit Rate
        SearchParams.Penalty = 300 * np.ones(np.shape(SearchParams.RevistRate))  # Penalty for exceeding UB
        SearchParams.Slope = 1. / SearchParams.RevistRate
        Nsearch = np.sum(SearchParams.NbeamsPerRow)
        SearchParams.JobDuration = np.array([])
        SearchParams.JobSlope = np.array([])
        SearchParams.DropTime = np.array([])  # Task dropping time. Will get updated as tasks get processed
        SearchParams.DropTimeFixed = np.array(
            [])  # Used to update DropTimes. Fixed for a given task e.x. always 2.6 process task at time 1 DropTime becomes 3.6
        SearchParams.DropCost = np.array([])
        for jj in range(len(SearchParams.NbeamsPerRow)):
            SearchParams.JobDuration = np.append(SearchParams.JobDuration,
                                                 np.repeat(SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.JobSlope = np.append(SearchParams.JobSlope,
                                              np.repeat(SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTime = np.append(SearchParams.DropTime,
                                              np.repeat(SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTimeFixed = np.append(SearchParams.DropTimeFixed, np.repeat(SearchParams.RevisitRateUB[jj],
                                                                                         SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropCost = np.append(SearchParams.DropCost,
                                              np.repeat(SearchParams.Penalty[jj], SearchParams.NbeamsPerRow[jj]))

        # %% Generate Track Tasks
        TrackParams = TaskParameters()  # Initializes to something like matlab structure
        # Ntrack = 10

        # Spawn tracks with uniformly distributed ranges and velocity
        MaxRangeNmi = 200  #
        MaxRangeRateMps = 343  # Mach 1 in Mps is 343

        truth = TaskParameters
        truth.rangeNmi = MaxRangeNmi * np.random.uniform(0, 1, Ntrack)
        truth.rangeRateMps = 2 * MaxRangeRateMps * np.random.uniform(0, 1, Ntrack) - MaxRangeRateMps

        TrackParams.DwellTime = np.array([18, 18, 18]) * 1e-3
        TrackParams.RevisitRate = np.array([1, 2, 4])
        TrackParams.RevisitRateUB = TrackParams.RevisitRate + 0.1
        TrackParams.Penalty = 300 * np.ones(np.shape(TrackParams.DwellTime))
        TrackParams.Slope = 1. / TrackParams.RevisitRate
        TrackParams.JobDuration = []
        TrackParams.JobSlope = []
        TrackParams.DropTime = []
        TrackParams.DropTimeFixed = []
        TrackParams.DropCost = []
        for jj in range(Ntrack):
            if truth.rangeNmi[jj] <= 50:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[0])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[0])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[0])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[0])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[0])
            elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[1])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[1])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
            else:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[2])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[2])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])

        # %% Begin Scheduler Loop

        # rng = np.random.default_rng(100)
        # task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
        #                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
        # tasks = task_gen.rand_tasks(N)

        # A = list()
        job = []
        cnt = 0  # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
        for ii in range(Nsearch):
            # job.append(0, ReluDropTask(SearchParams.JobDuration[ii], SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
            job.append(ReluDrop(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii],
                                SearchParams.DropCost[ii]))
            job[ii].Id = cnt  # Numeric Identifier for each job
            cnt = cnt + 1
            if job[ii].slope == 0.4:
                job[ii].Type = 'HS'  # Horizon Search (Used to determine revisit rates by job type
            else:
                job[ii].Type = 'AHS'  # Above horizon search
            job[ii].Priority = job[ii](0)  # Priority used to select which jobs to give to scheduler

            # tasks = ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
            # A.append(tasks)
            # del tasks
        for ii in range(Ntrack):
            # job.append(ReluDropTask(0, TrackParams.JobDuration[ii], TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
            job.append(ReluDrop(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii],
                                TrackParams.DropCost[ii]))
            job[cnt].Id = cnt  # Numeric Identifier for each job
            if job[cnt].slope == 0.25:
                job[cnt].Type = 'Tlow'  # Low Priority Track
            elif job[cnt].slope == 0.5:
                job[cnt].Type = 'Tmed'  # Medium Priority Track
            else:
                job[cnt].Type = 'Thigh'  # High Priority Track
            job[cnt].Priority = job[cnt](0)
            cnt = cnt + 1

        slope = np.array([task.slope for task in job])
        duration = np.array([task.duration for task in job])

        Capacity = np.sum(slope * np.round(
            duration / (RP / 2)) * RP / 2)  # Copied from matlab. Not sure why I divided by 2. Maybe 2 timelines.
        print(
            Capacity)  # Remembering. RP/2 has to do with FlexDAR tasks durations. They are either 18ms or 36 ms in this implementation. The RP is 40 ms. Therefore you can fit at most two jobs on the timeline, hence the 2

        ## Record Algorithm Performance
        # %% Evaluate
        # MaxTime = 20
        NumSteps = np.int(np.round(MaxTime / RP))

        self.NumSteps = NumSteps
        self.job = job
        self.Capacity = Capacity
        self.timeSec = np.array(0)
        self.cnt = 0
        self.iter = 0

        M = len(job)
        NUM_FEATS = 5  # Number of feautures
        features = np.empty((M, NUM_FEATS))
        feature_names = []
        feature_names.append('t_release-timeSec')
        feature_names.append('duration')
        feature_names.append('t_drop')
        feature_names.append('l_drop')
        feature_names.append('slope')
        for mm in range(len(job)):
            task = job[mm]
            features[mm, 0] = np.array([task.t_release] - self.timeSec)
            features[mm, 1] = np.array([task.duration])
            features[mm, 2] = np.array([task.t_drop])
            features[mm, 3] = np.array([task.l_drop])
            features[mm, 4] = np.array([task.slope])
            # features[mm, 5] = np.array(timeSec)
            # features[mm, 6] = np.array(ChannelAvailableTime)



        # self.paddle.goto(0, -275)
        # self.ball.goto(0, 100)
        # return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        # Reset ChannelAvailableTime
        K = self.env_config.get("K")
        ChannelAvailableTime = np.zeros(K)
        self.env_config['ChannelAvailableTime'] = ChannelAvailableTime


        self.M = M
        self.NUM_FEATS = NUM_FEATS
        self.done = False
        state = self.map_features_to_state(features)

        return state

    def step(self, action):

        timeSec = self.timeSec
        ChannelAvailableTime = self.env_config.get("ChannelAvailableTime")



        if np.min(ChannelAvailableTime) > timeSec:  # Go to next iteration
            state = self.state
            info = self.info
            job = self.job
            t_max = timeSec
            l_ex_remainder = eval_loss(job, t_max * np.ones(len(job)))  # Put all other tasks at max time and evaluate cost
            # reward = l_ex + l_ex_remainder
            reward = -l_ex_remainder  # TODO: Is this the right metric? Reward only a function of tasks not scheduled yet

            # RunTime[ii, idx_alg] = math.nan
            # Cost[ii, idx_alg] = math.nan
        else:

            # Pull-out local variables needed from class
            state = self.state
            info = self.info
            Job_Revisit_Count = info.get("Job_Revisit_Count")
            Job_Revisit_Time = info.get("Job_Revisit_Time")


            RP = self.env_config.get("RP")
            M = self.M
            N = self.env_config.get("N")
            K = self.env_config.get("K")
            NUM_FEATS = self.NUM_FEATS
            alg_func = self.env_config.get("scheduler")
            MaxTime = self.env_config.get("MaxTime")
            possible_actions = np.arange(0,M)
            job = self.job

            priority_Idx = np.empty(N,dtype='int64')
            for jj in range(N):
                priority_Idx[jj] = possible_actions[action[jj]]
                possible_actions= np.delete(possible_actions, action[jj])  # Remove previously taken actions

            job_scheduler = [] # Jobs to be scheduled (Length N)
            for nn in range(N):
                job_scheduler.append(job[priority_Idx[nn]]) # Copy desired job

            unwanted = priority_Idx[0:N]
            for ele in sorted(unwanted, reverse=True):
                del job[ele]

            # Schedule the Tasks
            t_start = time.time()
            t_ex, ch_ex = alg_func(job_scheduler, ChannelAvailableTime)  # Added Sequence T
            t_run = time.time() - t_start

            check_valid(job_scheduler, t_ex, ch_ex)
            l_ex = eval_loss(job_scheduler, t_ex)
            max_idx = np.argsort(t_ex)[-1]
            t_max = t_ex[max_idx] + job_scheduler[max_idx].duration
            l_ex_remainder = eval_loss(job, t_max * np.ones(len(job)))  # Put all other tasks at max time and evaluate cost
            # reward = l_ex + l_ex_remainder
            reward = -l_ex_remainder  # TODO: Is this the right metric? Reward only a function of tasks not scheduled yet


            # Logic to put tasks that are scheduled after RP back on job stack
            # Update ChannelAvailable Time
            duration = np.array([task.duration for task in job_scheduler])
            t_complete = t_ex + duration
            executed_tasks = t_complete <= timeSec + RP  # Task that are executed
            for kk in range(K):
                ChannelAvailableTime[kk] = np.max(t_complete[(ch_ex == kk) & executed_tasks])
            self.env_config["ChannelAvailableTime"] = ChannelAvailableTime

            # plot_results(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])
            # plot_loss_runtime(max_runtimes, l_ex_mean[i_gen], ax=ax_gen[1])
            for n in range(len(job_scheduler)):
                if executed_tasks[n]:  # Only updated executed tasks
                    job_scheduler[n].t_release = t_ex[n] + job_scheduler[
                        n].duration  # Update Release Times based on execution + duration
                    Job_Revisit_Count[job_scheduler[n].Id, 0] = Job_Revisit_Count[job_scheduler[n].Id, 0] + 1
                    Job_Revisit_Time[job_scheduler[n].Id][0].append(timeSec)
                job.append(job_scheduler[n])

            # Updates Features
            features = np.empty((M, NUM_FEATS))
            feature_names = []
            feature_names.append('t_release-timeSec')
            feature_names.append('duration')
            feature_names.append('t_drop')
            feature_names.append('l_drop')
            feature_names.append('slope')
            for mm in range(len(job)):
                task = job[mm]
                features[mm, 0] = np.array([task.t_release] - self.timeSec)
                features[mm, 1] = np.array([task.duration])
                features[mm, 2] = np.array([task.t_drop])
                features[mm, 3] = np.array([task.l_drop])
                features[mm, 4] = np.array([task.slope])

            state = self.map_features_to_state(features)  # Update State
            self.state = state

            if timeSec > MaxTime:
                self.done = True

            # self.done = 0
            info = {"Job_Revisit_Count": Job_Revisit_Count,
                    "Job_Revisit_Time": Job_Revisit_Time
                    }
            self.info = info

        self.reward = reward
        self.iter += 1
        self.timeSec = np.array(self.iter*RP)


        return state, self.reward, self.done, info


##
class TaskAssignmentDiscrete(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config=None):

        """
        N_track - number of tracks
        N_search is fixed to 120

        """
        if env_config:
            self.env_config = env_config
        else:
            alg_func = partial(earliest_release, do_swap=True)

            env_config = {"RP": 0.040,  # Resource Period
                          "MaxTime": 20,  # MaxTime to run an episode
                          "Ntrack": 40,  # Number of dedicated tracks
                          "N": 3,  # Number of jobs to process at any time
                          "K": 1,   # Number of Channels
                          "Nbins": 5, # Number of bins to discretize time variables
                          "scheduler": alg_func  # Function used to perform scheduling --> in the future use B&B or NN
                          }
            NumSteps = np.int(np.round(env_config.get("MaxTime") / env_config.get("RP")))
            ChannelAvailableTime = np.zeros(env_config.get("K"))
            env_config['NumSteps'] = NumSteps
            env_config['ChannelAvailableTime'] = ChannelAvailableTime

            self.env_config = env_config




        self.iter = 0

        N = self.env_config.get("N")
        actions = []
        for jj in range(N):
            actions.append(N-jj)

        self.action_space = gym.spaces.MultiDiscrete(actions)  # Works like this, first action pull off one job out of M, next action pull off one job out of (M-1) remaining, ..., repeat until N jobs have been pulled

        # self.action_space = Tuple((Discrete(2), Box(-10, 10, (2,))))
        # self.observation_space = Box(-10, 10, (2, 2))

        Nbins = self.env_config.get('Nbins')
        array_duration = 2*np.ones(N)
        array_t_drop = 5*np.ones(N)
        array_slope = 5*np.ones(N)
        array_t_release = Nbins*np.ones(N)
        array = np.hstack((array_t_release, array_duration, array_t_drop, array_slope))

        self.observation_space = gym.spaces.MultiDiscrete(array)
        # self.observation_space = gym.spaces.Tuple([Box(low=-12, high=1.0, shape=(M,)), MultiDiscrete(array_duration), MultiDiscrete(array_t_drop), MultiDiscrete(array_slope)])
        # self.observation_space = gym.spaces.Tuple([Box(-12, 0, (1,)), Discrete(2), Discrete(5), Discrete(4), (-10, 10, (2, 2))])


        state = self.reset()
        self.state = state

        ## Metrics
        Job_Revisit_Count = np.zeros((len(self.job), 1))
        Job_Revisit_Time = []
        for ii in range(len(self.job)):  # Create a list of empty lists --> make it multi-dimensional to support different algorithms
            elem = []
            for jj in range(1):
                elem.append([])
            Job_Revisit_Time.append(elem)

        info = {"Job_Revisit_Count": Job_Revisit_Count,
                "Job_Revisit_Time": Job_Revisit_Time
                }

        self.info = info


        # M = self.M



    def map_features_to_state(self, features, params):

        N = self.env_config.get('N')
        M = self.M
        NUM_FEATS = self.NUM_FEATS
        Nbins = self.env_config.get('Nbins')


        # Map features to observation space
        unique_duration = params.get('unique_duration')
        unique_t_drop = params.get('unique_t_drop')
        unique_slope = params.get('unique_slope')
        unique_t_release = params.get('unique_t_release')

        # unique_duration = np.unique(features[:, 1])
        # unique_t_drop = np.unique(features[:, 2])
        # unique_slope = np.unique(features[:, 4])
        # low = -12
        # high = 1
        # increment = (high-low)/Nbins
        # unique_t_release = np.empty(Nbins+1)
        # for jj in range(Nbins+1):
        #     unique_t_release[jj] = low + increment*jj

        temp = np.empty((N, NUM_FEATS - 1))
        # features_discretized = np.empty((N, NUM_FEATS-1))
        # temp[:, 0] = features[:, 0]
        for jj in range(Nbins):
            idx = np.where((features[:, 0] > unique_t_release[jj]) & (features[:, 0] < unique_t_release[jj+1]) )
            temp[idx, 0] = jj

        for jj in range(len(unique_duration)):
            idx = np.where(features[:, 1] == unique_duration[jj])
            temp[idx, 1] = jj

        for jj in range(len(unique_t_drop)):
            idx = np.where(features[:, 2] == unique_t_drop[jj])
            temp[idx, 2] = jj

        for jj in range(len(unique_slope)):
            idx = np.where(features[:, 4] == unique_slope[jj])
            temp[idx, 3] = jj

        # state = (np.array(temp[:, 0], dtype='f'), np.array(temp[:, 1], dtype='int64'),
        #          np.array(temp[:, 2], dtype='int64'),
        #          np.array(temp[:, 3], dtype='int64'))

        state = np.hstack((np.array(temp[:, 0], dtype='int64'), np.array(temp[:, 1], dtype='int64'),
                np.array(temp[:, 2], dtype='int64'),
                np.array(temp[:, 3], dtype='int64')))


        return state

    def map_state_to_features(self, state, params):

        N = self.env_config.get('N')
        M = self.M
        NUM_FEATS = self.NUM_FEATS
        Nbins = self.env_config.get('Nbins')

        # Map features to observation space
        unique_duration = params.get('unique_duration')
        unique_t_drop = params.get('unique_t_drop')
        unique_slope = params.get('unique_slope')
        unique_t_release = params.get('unique_t_release')

        values_list = list(params.values())

        features = np.empty((N, NUM_FEATS-1))  # Note NUM_FEATS is one less here because the dropping loss is not currently encoded.
        for ii in range(N):
            for jj in range(NUM_FEATS-1):  # Note NUM_FEATS is one less here because the dropping loss is not currently encoded.
                state_index = ii + jj*N
                feat_vector = values_list[jj]
                features[ii,jj] = feat_vector[state[state_index]]



        return features




    def reset(self):

        RP = self.env_config.get("RP")
        Ntrack = self.env_config.get("Ntrack")
        MaxTime = self.env_config.get("MaxTime")

        random.seed(30)


        ## Generate Search Tasks
        SearchParams = TaskParameters()
        SearchParams.NbeamsPerRow = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
        # SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload
        SearchParams.DwellTime = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18]) * 1e-3
        SearchParams.RevistRate = np.array([2.5, 5, 5, 5, 5, 5, 5, 5, 5])
        SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1  # Upper Bound on Revisit Rate
        SearchParams.Penalty = 300 * np.ones(np.shape(SearchParams.RevistRate))  # Penalty for exceeding UB
        SearchParams.Slope = 1. / SearchParams.RevistRate
        Nsearch = np.sum(SearchParams.NbeamsPerRow)
        SearchParams.JobDuration = np.array([])
        SearchParams.JobSlope = np.array([])
        SearchParams.DropTime = np.array([])  # Task dropping time. Will get updated as tasks get processed
        SearchParams.DropTimeFixed = np.array(
            [])  # Used to update DropTimes. Fixed for a given task e.x. always 2.6 process task at time 1 DropTime becomes 3.6
        SearchParams.DropCost = np.array([])
        for jj in range(len(SearchParams.NbeamsPerRow)):
            SearchParams.JobDuration = np.append(SearchParams.JobDuration,
                                                 np.repeat(SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.JobSlope = np.append(SearchParams.JobSlope,
                                              np.repeat(SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTime = np.append(SearchParams.DropTime,
                                              np.repeat(SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTimeFixed = np.append(SearchParams.DropTimeFixed, np.repeat(SearchParams.RevisitRateUB[jj],
                                                                                         SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropCost = np.append(SearchParams.DropCost,
                                              np.repeat(SearchParams.Penalty[jj], SearchParams.NbeamsPerRow[jj]))

        # %% Generate Track Tasks
        TrackParams = TaskParameters()  # Initializes to something like matlab structure
        # Ntrack = 10

        # Spawn tracks with uniformly distributed ranges and velocity
        MaxRangeNmi = 200  #
        MaxRangeRateMps = 343  # Mach 1 in Mps is 343

        truth = TaskParameters
        truth.rangeNmi = MaxRangeNmi * np.random.uniform(0, 1, Ntrack)
        truth.rangeRateMps = 2 * MaxRangeRateMps * np.random.uniform(0, 1, Ntrack) - MaxRangeRateMps

        TrackParams.DwellTime = np.array([18, 18, 18]) * 1e-3
        TrackParams.RevisitRate = np.array([1, 2, 4])
        TrackParams.RevisitRateUB = TrackParams.RevisitRate + 0.1
        TrackParams.Penalty = 300 * np.ones(np.shape(TrackParams.DwellTime))
        TrackParams.Slope = 1. / TrackParams.RevisitRate
        TrackParams.JobDuration = []
        TrackParams.JobSlope = []
        TrackParams.DropTime = []
        TrackParams.DropTimeFixed = []
        TrackParams.DropCost = []
        for jj in range(Ntrack):
            if truth.rangeNmi[jj] <= 50:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[0])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[0])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[0])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[0])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[0])
            elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[1])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[1])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
            else:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[2])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[2])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])

        # %% Begin Scheduler Loop

        # rng = np.random.default_rng(100)
        # task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
        #                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
        # tasks = task_gen.rand_tasks(N)

        # A = list()
        job = []
        cnt = 0  # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
        for ii in range(Nsearch):
            # job.append(0, ReluDropTask(SearchParams.JobDuration[ii], SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
            job.append(ReluDrop(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii],
                                SearchParams.DropCost[ii]))
            job[ii].Id = cnt  # Numeric Identifier for each job
            cnt = cnt + 1
            if job[ii].slope == 0.4:
                job[ii].Type = 'HS'  # Horizon Search (Used to determine revisit rates by job type
            else:
                job[ii].Type = 'AHS'  # Above horizon search
            job[ii].Priority = job[ii](0)  # Priority used to select which jobs to give to scheduler

            # tasks = ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
            # A.append(tasks)
            # del tasks
        for ii in range(Ntrack):
            # job.append(ReluDropTask(0, TrackParams.JobDuration[ii], TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
            job.append(ReluDrop(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii],
                                TrackParams.DropCost[ii]))
            job[cnt].Id = cnt  # Numeric Identifier for each job
            if job[cnt].slope == 0.25:
                job[cnt].Type = 'Tlow'  # Low Priority Track
            elif job[cnt].slope == 0.5:
                job[cnt].Type = 'Tmed'  # Medium Priority Track
            else:
                job[cnt].Type = 'Thigh'  # High Priority Track
            job[cnt].Priority = job[cnt](0)
            cnt = cnt + 1

        slope = np.array([task.slope for task in job])
        duration = np.array([task.duration for task in job])

        Capacity = np.sum(slope * np.round(
            duration / (RP / 2)) * RP / 2)  # Copied from matlab. Not sure why I divided by 2. Maybe 2 timelines.
        # print( Capacity)  # Remembering. RP/2 has to do with FlexDAR tasks durations. They are either 18ms or 36 ms in this implementation. The RP is 40 ms. Therefore you can fit at most two jobs on the timeline, hence the 2

        ## Record Algorithm Performance
        # %% Evaluate
        # MaxTime = 20
        NumSteps = np.int(np.round(MaxTime / RP))

        self.NumSteps = NumSteps
        self.Capacity = Capacity
        self.timeSec = np.array(0)
        self.cnt = 0
        self.iter = 0

        M = len(job)
        NUM_FEATS = 5  # Number of feautures
        features = np.empty((M, NUM_FEATS))
        feature_names = []
        feature_names.append('t_release-timeSec')
        feature_names.append('duration')
        feature_names.append('t_drop')
        feature_names.append('l_drop')
        feature_names.append('slope')
        for mm in range(len(job)):
            task = job[mm]
            # Generate new release time between 0 and 110% of t_drop
            low = -12
            high = 1
            t_new = np.random.uniform(low, high) # Taken from map_features_to_state function

            # t_new = np.random.uniform(np.array([task.t_drop])*1.1)

            # features[mm, 0] = np.array([task.t_release] - self.timeSec)
            features[mm, 0] = np.array(t_new)

            features[mm, 1] = np.array([task.duration])
            features[mm, 2] = np.array([task.t_drop])
            features[mm, 3] = np.array([task.l_drop])
            features[mm, 4] = np.array([task.slope])
            # features[mm, 5] = np.array(timeSec)
            # features[mm, 6] = np.array(ChannelAvailableTime)



        # self.paddle.goto(0, -275)
        # self.ball.goto(0, 100)
        # return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        # Reset ChannelAvailableTime
        K = self.env_config.get("K")
        ChannelAvailableTime = np.zeros(K)
        self.env_config['ChannelAvailableTime'] = ChannelAvailableTime


        self.M = M
        self.NUM_FEATS = NUM_FEATS
        # self.done = False
        N = self.env_config.get('N')

        # Choose N of M feature rows here
        feature_index = np.random.choice(np.arange(M), np.int(N), replace=False)
        N_features = features[feature_index,:]

        # Only keep relevant jobs. Needed later for scheduling purposes
        N_job = []
        for jj in range(N):
            N_job.append(job[feature_index[jj]])

        self.job = N_job



        # Map features to observation space
        params = {}
        low = -12
        high = 1
        Nbins = self.env_config.get('Nbins')
        increment = (high-low)/Nbins
        unique_t_release = np.empty(Nbins+1)
        for jj in range(Nbins+1):
            unique_t_release[jj] = low + increment*jj
        params['unique_t_release'] = unique_t_release
        params['unique_duration'] = np.unique(features[:, 1])
        params['unique_t_drop'] = np.unique(features[:, 2])
        params['unique_slope'] = np.unique(features[:, 4])


        self.env_config['params'] = params

        state = self.map_features_to_state(N_features, params)

        feat_check = self.map_state_to_features(state, params)


        return state

    def step(self, action):

        timeSec = self.timeSec
        ChannelAvailableTime = self.env_config.get("ChannelAvailableTime")



        if np.min(ChannelAvailableTime) > timeSec:  # Go to next iteration
            state = self.state
            info = self.info
            job = self.job
            t_max = timeSec
            l_ex_remainder = eval_loss(job, t_max * np.ones(len(job)))  # Put all other tasks at max time and evaluate cost
            # reward = l_ex + l_ex_remainder
            reward = -l_ex_remainder  # TODO: Is this the right metric? Reward only a function of tasks not scheduled yet

            # RunTime[ii, idx_alg] = math.nan
            # Cost[ii, idx_alg] = math.nan
        else:

            # Pull-out local variables needed from class
            state = self.state
            info = self.info
            Job_Revisit_Count = info.get("Job_Revisit_Count")
            Job_Revisit_Time = info.get("Job_Revisit_Time")


            RP = self.env_config.get("RP")
            M = self.M
            N = self.env_config.get("N")
            K = self.env_config.get("K")
            NUM_FEATS = self.NUM_FEATS
            alg_func = self.env_config.get("scheduler")
            MaxTime = self.env_config.get("MaxTime")
            possible_actions = np.arange(0,N)
            job = self.job

            priority_Idx = np.empty(N, dtype='int64')
            for jj in range(N):
                priority_Idx[jj] = possible_actions[action[jj]]
                possible_actions = np.delete(possible_actions, action[jj])  # Remove previously taken actions

            job_scheduler = []  # Jobs to be scheduled (Length N)
            for nn in range(N):
                job_scheduler.append(job[priority_Idx[nn]]) # Copy desired job

            # unwanted = priority_Idx[0:N]
            # for ele in sorted(unwanted, reverse=True):
            #     del job[ele]

            # Schedule the Tasks
            t_start = time.time()
            t_ex, ch_ex = alg_func(job_scheduler, ChannelAvailableTime)  # Added Sequence T
            t_run = time.time() - t_start

            check_valid(job_scheduler, t_ex, ch_ex)
            l_ex = eval_loss(job_scheduler, t_ex)
            # max_idx = np.argsort(t_ex)[-1]
            # t_max = t_ex[max_idx] + job_scheduler[max_idx].duration
            # l_ex_remainder = evaluate_schedule(job, t_max * np.ones(len(job)))  # Put all other tasks at max time and evaluate cost
            # reward = l_ex + l_ex_remainder
            reward = -l_ex  # TODO: Is this the right metric? Reward only a function of tasks not scheduled yet


            # # Logic to put tasks that are scheduled after RP back on job stack
            # # Update ChannelAvailable Time
            # duration = np.array([task.duration for task in job_scheduler])
            # t_complete = t_ex + duration
            # executed_tasks = t_complete <= timeSec + RP  # Task that are executed
            # for kk in range(K):
            #     ChannelAvailableTime[kk] = np.max(t_complete[(ch_ex == kk) & executed_tasks])
            # self.env_config["ChannelAvailableTime"] = ChannelAvailableTime
            #
            # # plot_results(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])
            # # plot_loss_runtime(max_runtimes, l_ex_mean[i_gen], ax=ax_gen[1])
            # for n in range(len(job_scheduler)):
            #     if executed_tasks[n]:  # Only updated executed tasks
            #         job_scheduler[n].t_release = t_ex[n] + job_scheduler[
            #             n].duration  # Update Release Times based on execution + duration
            #         Job_Revisit_Count[job_scheduler[n].Id, 0] = Job_Revisit_Count[job_scheduler[n].Id, 0] + 1
            #         Job_Revisit_Time[job_scheduler[n].Id][0].append(timeSec)
            #     job.append(job_scheduler[n])
            #
            # # Updates Features
            # features = np.empty((M, NUM_FEATS))
            # feature_names = []
            # feature_names.append('t_release-timeSec')
            # feature_names.append('duration')
            # feature_names.append('t_drop')
            # feature_names.append('l_drop')
            # feature_names.append('slope')
            # for mm in range(len(job)):
            #     task = job[mm]
            #     features[mm, 0] = np.array([task.t_release] - self.timeSec)
            #     features[mm, 1] = np.array([task.duration])
            #     features[mm, 2] = np.array([task.t_drop])
            #     features[mm, 3] = np.array([task.l_drop])
            #     features[mm, 4] = np.array([task.slope])
            #
            # state = self.map_features_to_state(features)  # Update State
            # self.state = state
            #
            # if timeSec > MaxTime:
            #     self.done = True
            #
            # # self.done = 0
            info = {"Job_Revisit_Count": Job_Revisit_Count,
                    "Job_Revisit_Time": Job_Revisit_Time
                    }
            self.info = info

        state = self.state
        done = True
        # self.reset()
        self.reward = reward
        self.iter += 1
        self.timeSec = np.array(self.iter*RP)


        return state, reward, done, info

    def close(self):
        plt.close('all')


















class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)


class PriorityQueue(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config=None):

        """
        N_track - number of tracks
        N_search is fixed to 120

        """
        if env_config:
            self.env_config = env_config
        else:
            alg_func = partial(earliest_release, do_swap=True)

            env_config = {"RP": 0.040,  # Resource Period
                          "MaxTime": 20,  # MaxTime to run an episode
                          "Ntrack": 40,  # Number of dedicated tracks
                          "N": 8,  # Number of jobs to process at any time
                          "K": 1,   # Number of Channels
                          "scheduler": alg_func  # Function used to perform scheduling --> in the future use B&B or NN
                          }
            NumSteps = np.int(np.round(env_config.get("MaxTime") / env_config.get("RP")))
            ChannelAvailableTime = np.zeros(env_config.get("K"))
            env_config['NumSteps'] = NumSteps
            env_config['ChannelAvailableTime'] = ChannelAvailableTime

            self.env_config = env_config



        state = self.reset()
        self.state = state

        N = self.env_config.get("N")
        M = self.M
        actions = []
        for jj in range(N):
            actions.append(M-jj)

        self.action_space = gym.spaces.MultiDiscrete(actions)  # Works like this, first action pull off one job out of M, next action pull off one job out of (M-1) remaining, ..., repeat until N jobs have been pulled

        # self.action_space = Tuple((Discrete(2), Box(-10, 10, (2,))))
        # self.observation_space = Box(-10, 10, (2, 2))

        array_duration = 2*np.ones(M)
        array_t_drop = 5*np.ones(M)
        array_slope = 5*np.ones(M)

        self.observation_space = gym.spaces.Tuple([Box(low=-12, high=1.0, shape=(M,)), MultiDiscrete(array_duration), MultiDiscrete(array_t_drop), MultiDiscrete(array_slope)])
        # self.observation_space = gym.spaces.Tuple([Box(-12, 0, (1,)), Discrete(2), Discrete(5), Discrete(4), (-10, 10, (2, 2))])

        ## Metrics
        Job_Revisit_Count = np.zeros((len(self.job), 1))
        Job_Revisit_Time = []
        for ii in range(len(self.job)):  # Create a list of empty lists --> make it multi-dimensional to support different algorithms
            elem = []
            for jj in range(1):
                elem.append([])
            Job_Revisit_Time.append(elem)

        info = {"Job_Revisit_Count": Job_Revisit_Count,
                "Job_Revisit_Time": Job_Revisit_Time
                }

        self.info = info



    def map_features_to_state(self, features):

        M = self.M
        NUM_FEATS = self.NUM_FEATS

        # Map features to observation space
        unique_duration = np.unique(features[:, 1])
        unique_t_drop = np.unique(features[:, 2])
        unique_slope = np.unique(features[:, 4])

        temp = np.empty((M, NUM_FEATS - 1))
        temp[:, 0] = features[:, 0]
        for jj in range(len(unique_duration)):
            idx = np.where(features[:, 1] == unique_duration[jj])
            temp[idx, 1] = jj

        for jj in range(len(unique_t_drop)):
            idx = np.where(features[:, 2] == unique_t_drop[jj])
            temp[idx, 2] = jj

        for jj in range(len(unique_slope)):
            idx = np.where(features[:, 4] == unique_slope[jj])
            temp[idx, 3] = jj

        state = (np.array(temp[:, 0], dtype='f'), np.array(temp[:, 1], dtype='int64'),
                 np.array(temp[:, 2], dtype='int64'),
                 np.array(temp[:, 3], dtype='int64'))


        return state



    def reset(self):

        RP = self.env_config.get("RP")
        Ntrack = self.env_config.get("Ntrack")
        MaxTime = self.env_config.get("MaxTime")

        random.seed(30)


        ## Generate Search Tasks
        SearchParams = TaskParameters()
        SearchParams.NbeamsPerRow = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
        # SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload
        SearchParams.DwellTime = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18]) * 1e-3
        SearchParams.RevistRate = np.array([2.5, 5, 5, 5, 5, 5, 5, 5, 5])
        SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1  # Upper Bound on Revisit Rate
        SearchParams.Penalty = 300 * np.ones(np.shape(SearchParams.RevistRate))  # Penalty for exceeding UB
        SearchParams.Slope = 1. / SearchParams.RevistRate
        Nsearch = np.sum(SearchParams.NbeamsPerRow)
        SearchParams.JobDuration = np.array([])
        SearchParams.JobSlope = np.array([])
        SearchParams.DropTime = np.array([])  # Task dropping time. Will get updated as tasks get processed
        SearchParams.DropTimeFixed = np.array(
            [])  # Used to update DropTimes. Fixed for a given task e.x. always 2.6 process task at time 1 DropTime becomes 3.6
        SearchParams.DropCost = np.array([])
        for jj in range(len(SearchParams.NbeamsPerRow)):
            SearchParams.JobDuration = np.append(SearchParams.JobDuration,
                                                 np.repeat(SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.JobSlope = np.append(SearchParams.JobSlope,
                                              np.repeat(SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTime = np.append(SearchParams.DropTime,
                                              np.repeat(SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropTimeFixed = np.append(SearchParams.DropTimeFixed, np.repeat(SearchParams.RevisitRateUB[jj],
                                                                                         SearchParams.NbeamsPerRow[jj]))
            SearchParams.DropCost = np.append(SearchParams.DropCost,
                                              np.repeat(SearchParams.Penalty[jj], SearchParams.NbeamsPerRow[jj]))

        # %% Generate Track Tasks
        TrackParams = TaskParameters()  # Initializes to something like matlab structure
        # Ntrack = 10

        # Spawn tracks with uniformly distributed ranges and velocity
        MaxRangeNmi = 200  #
        MaxRangeRateMps = 343  # Mach 1 in Mps is 343

        truth = TaskParameters
        truth.rangeNmi = MaxRangeNmi * np.random.uniform(0, 1, Ntrack)
        truth.rangeRateMps = 2 * MaxRangeRateMps * np.random.uniform(0, 1, Ntrack) - MaxRangeRateMps

        TrackParams.DwellTime = np.array([18, 18, 18]) * 1e-3
        TrackParams.RevisitRate = np.array([1, 2, 4])
        TrackParams.RevisitRateUB = TrackParams.RevisitRate + 0.1
        TrackParams.Penalty = 300 * np.ones(np.shape(TrackParams.DwellTime))
        TrackParams.Slope = 1. / TrackParams.RevisitRate
        TrackParams.JobDuration = []
        TrackParams.JobSlope = []
        TrackParams.DropTime = []
        TrackParams.DropTimeFixed = []
        TrackParams.DropCost = []
        for jj in range(Ntrack):
            if truth.rangeNmi[jj] <= 50:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[0])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[0])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[0])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[0])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[0])
            elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[1])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[1])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
            else:
                TrackParams.JobDuration = np.append(TrackParams.JobDuration, TrackParams.DwellTime[2])
                TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
                TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
                TrackParams.DropTimeFixed = np.append(TrackParams.DropTimeFixed, TrackParams.RevisitRateUB[2])
                TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])

        # %% Begin Scheduler Loop

        # rng = np.random.default_rng(100)
        # task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
        #                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
        # tasks = task_gen.rand_tasks(N)

        # A = list()
        job = []
        cnt = 0  # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
        for ii in range(Nsearch):
            # job.append(0, ReluDropTask(SearchParams.JobDuration[ii], SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
            job.append(ReluDrop(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii],
                                SearchParams.DropCost[ii]))
            job[ii].Id = cnt  # Numeric Identifier for each job
            cnt = cnt + 1
            if job[ii].slope == 0.4:
                job[ii].Type = 'HS'  # Horizon Search (Used to determine revisit rates by job type
            else:
                job[ii].Type = 'AHS'  # Above horizon search
            job[ii].Priority = job[ii](0)  # Priority used to select which jobs to give to scheduler

            # tasks = ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
            # A.append(tasks)
            # del tasks
        for ii in range(Ntrack):
            # job.append(ReluDropTask(0, TrackParams.JobDuration[ii], TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
            job.append(ReluDrop(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii],
                                TrackParams.DropCost[ii]))
            job[cnt].Id = cnt  # Numeric Identifier for each job
            if job[cnt].slope == 0.25:
                job[cnt].Type = 'Tlow'  # Low Priority Track
            elif job[cnt].slope == 0.5:
                job[cnt].Type = 'Tmed'  # Medium Priority Track
            else:
                job[cnt].Type = 'Thigh'  # High Priority Track
            job[cnt].Priority = job[cnt](0)
            cnt = cnt + 1

        slope = np.array([task.slope for task in job])
        duration = np.array([task.duration for task in job])

        Capacity = np.sum(slope * np.round(
            duration / (RP / 2)) * RP / 2)  # Copied from matlab. Not sure why I divided by 2. Maybe 2 timelines.
        print(
            Capacity)  # Remembering. RP/2 has to do with FlexDAR tasks durations. They are either 18ms or 36 ms in this implementation. The RP is 40 ms. Therefore you can fit at most two jobs on the timeline, hence the 2

        ## Record Algorithm Performance
        # %% Evaluate
        # MaxTime = 20
        NumSteps = np.int(np.round(MaxTime / RP))

        self.NumSteps = NumSteps
        self.job = job
        self.Capacity = Capacity
        self.timeSec = np.array(0)
        self.cnt = 0

        M = len(job)
        NUM_FEATS = 5  # Number of feautures
        features = np.empty((M, NUM_FEATS))
        feature_names = []
        feature_names.append('t_release-timeSec')
        feature_names.append('duration')
        feature_names.append('t_drop')
        feature_names.append('l_drop')
        feature_names.append('slope')
        for mm in range(len(job)):
            task = job[mm]
            features[mm, 0] = np.array([task.t_release] - self.timeSec)
            features[mm, 1] = np.array([task.duration])
            features[mm, 2] = np.array([task.t_drop])
            features[mm, 3] = np.array([task.l_drop])
            features[mm, 4] = np.array([task.slope])
            # features[mm, 5] = np.array(timeSec)
            # features[mm, 6] = np.array(ChannelAvailableTime)



        # self.paddle.goto(0, -275)
        # self.ball.goto(0, 100)
        # return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        # Reset ChannelAvailableTime
        K = self.env_config.get("K")
        ChannelAvailableTime = np.zeros(K)
        self.env_config['ChannelAvailableTime'] = ChannelAvailableTime


        self.M = M
        self.NUM_FEATS = NUM_FEATS
        self.done = False
        state = self.map_features_to_state(features)


        return state

    def step(self, action):

        # Pull-out local variables needed from class
        state = self.state
        info = self.info
        Job_Revisit_Count = info.get("Job_Revisit_Count")
        Job_Revisit_Time = info.get("Job_Revisit_Time")


        timeSec = self.timeSec
        RP = self.env_config.get("RP")
        M = self.M
        N = self.env_config.get("N")
        K = self.env_config.get("K")
        NUM_FEATS = self.NUM_FEATS
        alg_func = self.env_config.get("scheduler")
        ChannelAvailableTime = self.env_config.get("ChannelAvailableTime")
        MaxTime = self.env_config.get("MaxTime")
        possible_actions = np.arange(0,M)
        job = self.job

        priority_Idx = np.empty(N,dtype='int64')
        for jj in range(N):
            priority_Idx[jj] = possible_actions[action[jj]]
            possible_actions= np.delete(possible_actions, action[jj])  # Remove previously taken actions

        job_scheduler = [] # Jobs to be scheduled (Length N)
        for nn in range(N):
            job_scheduler.append(job[priority_Idx[nn]]) # Copy desired job

        unwanted = priority_Idx[0:N]
        for ele in sorted(unwanted, reverse=True):
            del job[ele]

        # Schedule the Tasks
        t_start = time.time()
        t_ex, ch_ex = alg_func(job_scheduler, ChannelAvailableTime)  # Added Sequence T
        t_run = time.time() - t_start

        check_valid(job_scheduler, t_ex, ch_ex)
        l_ex = eval_loss(job_scheduler, t_ex)
        max_idx = np.argsort(t_ex)[-1]
        t_max = t_ex[max_idx] + job_scheduler[max_idx].duration
        l_ex_remainder = eval_loss(job, t_max * np.ones(len(job)))  # Put all other tasks at max time and evaluate cost
        # reward = l_ex + l_ex_remainder
        reward = -l_ex_remainder  # TODO: Is this the right metric? Reward only a function of tasks not scheduled yet


        # Logic to put tasks that are scheduled after RP back on job stack
        # Update ChannelAvailable Time
        duration = np.array([task.duration for task in job_scheduler])
        t_complete = t_ex + duration
        executed_tasks = t_complete <= timeSec + RP  # Task that are executed
        for kk in range(K):
            ChannelAvailableTime[kk] = np.max(t_complete[(ch_ex == kk) & executed_tasks])
        self.env_config["ChannelAvailableTime"] = ChannelAvailableTime

        # plot_results(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])
        # plot_loss_runtime(max_runtimes, l_ex_mean[i_gen], ax=ax_gen[1])
        for n in range(len(job_scheduler)):
            if executed_tasks[n]:  # Only updated executed tasks
                job_scheduler[n].t_release = t_ex[n] + job_scheduler[
                    n].duration  # Update Release Times based on execution + duration
                Job_Revisit_Count[job_scheduler[n].Id, 0] = Job_Revisit_Count[job_scheduler[n].Id, 0] + 1
                Job_Revisit_Time[job_scheduler[n].Id][0].append(timeSec)
            job.append(job_scheduler[n])

        # Updates Features
        features = np.empty((M, NUM_FEATS))
        feature_names = []
        feature_names.append('t_release-timeSec')
        feature_names.append('duration')
        feature_names.append('t_drop')
        feature_names.append('l_drop')
        feature_names.append('slope')
        for mm in range(len(job)):
            task = job[mm]
            features[mm, 0] = np.array([task.t_release] - self.timeSec)
            features[mm, 1] = np.array([task.duration])
            features[mm, 2] = np.array([task.t_drop])
            features[mm, 3] = np.array([task.l_drop])
            features[mm, 4] = np.array([task.slope])

        state = self.map_features_to_state(features)  # Update State
        self.state = state

        if timeSec > MaxTime:
            self.done = True

        # self.done = 0
        self.reward = reward
        info = {"Job_Revisit_Count": Job_Revisit_Count,
                "Job_Revisit_Time": Job_Revisit_Time
                }
        self.info = info

        return state, self.reward, self.done, info



class JobScheduler():
    """
       Base environment for task scheduling.

       Parameters
       ----------
       N_track : int
           Number of track tasks.
       task_gen : GenericTaskGenerator
           Task generation object.
       n_ch: int
           Number of channels.
       ch_avail_gen : callable
           Returns random initial channel availabilities.
       node_cls : TreeNode or callable
           Class for tree search node generation.
       features : ndarray, optional
           Structured numpy array of features with fields 'name', 'func', and 'lims'.
       sort_func : function or str, optional
           Method that returns a sorting value for re-indexing given a task index 'n'.

       """






import turtle as t





class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self):

        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):

        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .1

        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done


# ------------------------ Human control ------------------------
#
# env = Paddle()
#
# while True:
#      env.run_frame()

class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self):

        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):

        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .1

        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done


# ------------------------ Human control ------------------------
#
# env = Paddle()
#
# while True:
#      env.run_frame()



# Env that models a 1D corridor (you can move left or right)
# Goal is to get to the end (i.e. move right [length] number of times)
# Inspired by Ray's RLlib example environment: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
# See also Sutton and Barto (e.g. example 13.1 on page 323).


class CorridorEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # Check [env_config] for corridor length, default to 10
    def __init__(self, env_config=None):
        self.env_config = env_config if env_config else {}
        self.length = int(self.env_config.get("length", 10))
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(self.length + 1)
        self.reset()

    def step(self, action):
        # =========
        # TODO - Implement a 1D corridor environment:
        #   * self.position is the agent's current position in the env
        #   * agent starts in self.position 0 (far left)
        #   * agent must walk right to the end of the corridor (self.length)
        #   * action is either a 0 (move left) or 1 (move right)
        #   * the agent hits wall if it walks left at self.position == 0
        #   * the agent succeeds when self.position == self.length
        #   * each step should award the agent -1 until goal is reached
        #   * Returns a 4-tuple: (new state, reward, done, info)
        # =========
        pass

    @property
    def done(self):
        return self.position >= self.length

    def reset(self):
        self.position = 0
        return self.position

    def render(self, mode="human"):
        pass

    def close(self):
        pass
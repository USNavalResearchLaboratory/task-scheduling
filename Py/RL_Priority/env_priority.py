import numpy as np
import matplotlib.pyplot as plt
import time     # TODO: use builtin module timeit instead? or cProfile?
import random
import math
from SL_policy_Discrete import load_policy, wrap_policy
from tree_search import branch_bound, random_sequencer, earliest_release
from functools import partial
from util.generic import algorithm_repr, check_rng
from util.plot import plot_task_losses, plot_schedule, plot_loss_runtime, scatter_loss_runtime
from util.results import check_valid, eval_loss
from more_itertools import locate
from math import factorial, floor
# import numpy as np
from tasks import ReluDropGenerator
# from tasks import ReluDropTask
from task_scheduling.tasks import ReluDrop
import gym
from gym.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete

##
class TaskParameters: # Initializes to something like matlab structure. Enables dot indexing
    pass

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
            env_config = {"RP": 0.040,  # Resource Period
                          "MaxTime": 10,  # MaxTime to run an episode
                          "Ntrack": 10,  # Number of dedicated tracks
                          "N": 8,  # Number of jobs to process at any time
                          "K": 1   # Number of Channels
                          }
            self.env_config = env_config



        state = self.reset()

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
        array_slope = 4*np.ones(M)

        self.observation_space = gym.spaces.Tuple([Box(low=-12, high=0.01, shape=(M,)), MultiDiscrete(array_duration), MultiDiscrete(array_t_drop), MultiDiscrete(array_slope)])
        # self.observation_space = gym.spaces.Tuple([Box(-12, 0, (1,)), Discrete(2), Discrete(5), Discrete(4), (-10, 10, (2, 2))])

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
                 np.array(temp[:, 3],dtype='int64'))


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
        self.M = M
        self.NUM_FEATS = NUM_FEATS
        state = self.map_features_to_state(features)


        return state

    def step(self, action):

        self.done = 0
        self.reward = 0
        state = 0

        return self.reward, state, self.done



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
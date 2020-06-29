def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step



import numpy as np
import matplotlib.pyplot as plt
import time     # TODO: use builtin module timeit instead? or cProfile?


#%% Set up Problem
N = 8 # Number of Jobs to process simultaneously
K = 2 # Number of timelines (radars)
RP = 0.04
Tmax = 50

## Specify Algorithms
from tree_search import branch_bound, random_sequencer, ert_alg_kw, est_alg_kw
from functools import partial
from util.generic import algorithm_repr, check_rng
from util.plot import plot_task_losses, plot_schedule, scatter_loss_runtime
from util.results import check_valid, eval_loss

from math import factorial, floor

alg_funcs = [partial(ert_alg_kw, do_swap=True)]
             #partial(branch_bound, verbose=False),
             # partial(mc_tree_search, n_mc=[floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)], verbose=False),
             # partial(random_sequencer),
             # partial(random_agent)]
alg_n_runs = [1]       # number of runs per problem
alg_reprs = list(map(algorithm_repr, alg_funcs))



##
class TaskParameters: # Initializes to something like matlab structure. Enables dot indexing
    pass

## Generate Search Tasks
SearchParams = TaskParameters()
SearchParams.NbeamsPerRow = np.array([28, 29, 14, 9, 10, 9, 8, 7, 6])
# SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload
SearchParams.DwellTime = np.array([36, 36, 36, 18, 18, 18, 18, 18, 18])*1e-3
SearchParams.RevistRate = np.array([2.5, 5, 5, 5, 5, 5, 5, 5, 5])
SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1 # Upper Bound on Revisit Rate
SearchParams.Penalty = 100*np.ones(np.shape(SearchParams.RevistRate)) # Penalty for exceeding UB
SearchParams.Slope = 1./SearchParams.RevistRate
Nsearch = np.sum(SearchParams.NbeamsPerRow)
SearchParams.JobDuration = np.array([])
SearchParams.JobSlope = np.array([])
SearchParams.DropTime = np.array([])
SearchParams.DropCost = np.array([])
for jj in range(len(SearchParams.NbeamsPerRow)):
    SearchParams.JobDuration = np.append(SearchParams.JobDuration, np.repeat( SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.JobSlope = np.append(SearchParams.JobSlope, np.repeat( SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.DropTime = np.append(SearchParams.DropTime, np.repeat( SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.DropCost = np.append(SearchParams.DropCost, np.repeat( SearchParams.Penalty[jj], SearchParams.NbeamsPerRow[jj]))

#%% Generate Track Tasks
TrackParams = TaskParameters() # Initializes to something like matlab structure
Ntrack = 10

# Spawn tracks with uniformly distributed ranges and velocity
MaxRangeNmi = 200 #
MaxRangeRateMps = 343 # Mach 1 in Mps is 343

truth = TaskParameters
truth.rangeNmi = MaxRangeNmi*np.random.uniform(0,1,Ntrack)
truth.rangeRateMps = 2*MaxRangeRateMps*np.random.uniform(0,1,Ntrack) - MaxRangeRateMps

TrackParams.DwellTime = np.array([18, 18, 18])*1e-3
TrackParams.RevisitRate = np.array([1, 2, 4])
TrackParams.RevisitRateUB = TrackParams.RevisitRate  + 0.1
TrackParams.Penalty = 300*np.ones(np.shape(TrackParams.DwellTime))
TrackParams.Slope = 1./TrackParams.RevisitRate
TrackParams.JobDuration = []
TrackParams.JobSlope = []
TrackParams.DropTime = []
TrackParams.DropCost = []
for jj in range(Ntrack):
    if  truth.rangeNmi[jj] <= 50:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[0]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope,  TrackParams.Slope[0] )
        TrackParams.DropTime = np.append( TrackParams.DropTime,   TrackParams.RevisitRateUB[0])
        TrackParams.DropCost = np.append( TrackParams.DropCost,   TrackParams.Penalty[0] )
    elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[1]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
        TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
        TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
    else:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[2]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
        TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
        TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])





#%% Begin Scheduler Loop




# import numpy as np
from tasks import ReluDropGenerator
from tasks import ReluDropTask

# rng = np.random.default_rng(100)
# task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
#                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
# tasks = task_gen.rand_tasks(N)

# A = list()
job = []
cnt = 1
for ii in range(Nsearch):
    job.append(ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii]))
    job[ii].Id = cnt # Numeric Identifier for each job
    cnt = cnt + 1
    if job[ii].slope == 0.4:
        job[ii].Type = 'HS' # Horizon Search (Used to determine revisit rates by job type
    else:
        job[ii].Type = 'AHS' # Above horizon search
    job[ii].Priority = job[ii].loss_fcn(0) # Priority used to select which jobs to give to scheduler

    # tasks = ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
    # A.append(tasks)
    # del tasks
for ii in range(Ntrack):
    job.append(ReluDropTask(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropCost[ii]))
    job[ii].Id = cnt # Numeric Identifier for each job
    cnt = cnt + 1
    if job[ii].slope == 0.25:
        job[ii].Type = 'Tlow' # Low Priority Track
    elif job[ii].slope == 0.5:
        job[ii].Type = 'Tmed' # Medium Priority Track
    else:
        job[ii].Type = 'Thigh' # High Priority Track
    job[ii].Priority = job[ii].loss_fcn(0)


slope = np.array([task.slope for task in job])
duration = np.array([task.duration for task in job])

Capacity = np.sum( slope*np.round(duration/(RP/2))*RP/2 ) # Copied from matlab. Not sure why I divided by 2. Maybe 2 timelines.
print(Capacity) # Remembering. RP/2 has to do with FlexDAR tasks durations. They are either 18ms or 36 ms in this implementation. The RP is 40 ms. Therefore you can fit at most two jobs on the timeline, hence the 2
a = 1

## Record Algorithm Performance
# %% Evaluate
MaxTime = 10
NumSteps = np.int(np.round(MaxTime/RP))
t_run_iter = np.array(list(zip(*[np.empty((NumSteps, n_run)) for n_run in alg_n_runs])),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

l_ex_iter = np.array(list(zip(*[np.empty((NumSteps, n_run)) for n_run in alg_n_runs])),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

t_run_mean = np.array(list(zip(*np.empty((len(alg_reprs), NumSteps)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), NumSteps)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))


## Begin Main Loop

for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
    for i_run in range(n_run):  # Perform new algorithm runs: Should never be more than 1 in this code

        ChannelAvailableTime = np.zeros(K)
        for ii in np.arange(NumSteps): # Main Loop to evaluate schedulers
            timeSec = ii*RP # Current time

            if np.min(ChannelAvailableTime) > timeSec:
                continue # Jump to next Resource Period

            # Reassess Track Priorities
            for jj in range(len(job)):
                job[jj].Priority = job[jj].loss_fcn(timeSec)

            priority = np.array([task.Priority for task in job])
            priority_Idx = np.argsort(priority)

            job_scheduler = [] # Jobs to be scheduled (Length N)
            for nn in range(N):
                job_scheduler.append(job.pop(priority_Idx[nn]))

            _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {1}', clear=True)
            plot_task_losses(job_scheduler, ax=ax_gen[0])


            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex, T = alg_func(job_scheduler, ChannelAvailableTime) # Added Sequence T
            t_run = time.time() - t_start

            check_valid(job_scheduler, t_ex, ch_ex)
            l_ex = eval_loss(job_scheduler, t_ex)

            t_run_iter[alg_repr][ii, i_run] = t_run
            l_ex_iter[alg_repr][ii, i_run] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

            t_run_mean[alg_repr][ii] = t_run_iter[alg_repr][ii].mean()
            l_ex_mean[alg_repr][ii] = l_ex_iter[alg_repr][ii].mean()

            print('')
            print(f"    Avg. Runtime: {t_run_mean[alg_repr][ii]:.2f} (s)")
            print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][ii]:.2f}")

            scatter_loss_runtime(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])

            # for n = indexExecution:
            #     new_job(n).Id = queue((n)).Id;
            #     new_job(n).StartTime = t_ex((n)) + queue((n)).Duration;
            #     new_job(n).slope = queue((n)).slope;
            #     new_job(n).DropTime = queue((n)).DropTime;
            #     new_job(n).DropRelativeTime = queue((n)).DropTime + new_job(n).StartTime; % Update with new start time and job DropTime
            #     new_job(n).DropCost = queue((n)).DropCost;
            #     new_job(n).Duration = queue((n)).Duration;
            #     new_job(n).Type = queue((n)).Type;
                
                # metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
                # JobRevistTime{queue(n).Id}(metrics.JobRevistCount(queue(n).Id)) = timeSec;

            # TODO Put jobs in job_scheduler at the end of the master list "job", Finish plotting









class Scheduler:

    def __init__(self):
        self.Id = 0
        self.slope = []
        self.StartTime = 0
        self.DropTime = []
        self.DropRelativeTime = []
        self.DropCost = 0
        self.Duration = 0
        self.Type = []
        self.Priority =0

        # job = struct('Id', 0, 'slope', [], 'StartTime', 0, 'DropTime', [], 'DropRelativeTime', [], 'DropCost', 0,
        #              'Duration', 0, 'Type', [], 'Priority', 0); % Place


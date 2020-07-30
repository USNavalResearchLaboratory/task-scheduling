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
# Tmax = 3

## Specify Algorithms
from radar_scheduling.tree_search import earliest_release
from functools import partial
from radar_scheduling.util.generic import algorithm_repr
from radar_scheduling.util.plot import plot_task_losses, scatter_loss_runtime
from radar_scheduling.util.results import check_valid, eval_loss

alg_funcs = [partial(earliest_release, do_swap=True)]
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
SearchParams.DropTime = np.array([])            # Task dropping time. Will get updated as tasks get processed
SearchParams.DropTimeFixed = np.array([])       # Used to update DropTimes. Fixed for a given task e.x. always 2.6 process task at time 1 DropTime becomes 3.6
SearchParams.DropCost = np.array([])
for jj in range(len(SearchParams.NbeamsPerRow)):
    SearchParams.JobDuration = np.append(SearchParams.JobDuration, np.repeat( SearchParams.DwellTime[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.JobSlope = np.append(SearchParams.JobSlope, np.repeat( SearchParams.Slope[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.DropTime = np.append(SearchParams.DropTime, np.repeat( SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
    SearchParams.DropTimeFixed = np.append(SearchParams.DropTimeFixed, np.repeat( SearchParams.RevisitRateUB[jj], SearchParams.NbeamsPerRow[jj]))
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
TrackParams.DropTimeFixed = []
TrackParams.DropCost = []
for jj in range(Ntrack):
    if  truth.rangeNmi[jj] <= 50:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[0]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope,  TrackParams.Slope[0] )
        TrackParams.DropTime = np.append( TrackParams.DropTime,   TrackParams.RevisitRateUB[0])
        TrackParams.DropTimeFixed = np.append( TrackParams.DropTimeFixed,   TrackParams.RevisitRateUB[0])
        TrackParams.DropCost = np.append( TrackParams.DropCost,   TrackParams.Penalty[0] )
    elif truth.rangeNmi[jj] > 50 and abs(truth.rangeRateMps[jj]) >= 100:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[1]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[1])
        TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[1])
        TrackParams.DropTimeFixed = np.append( TrackParams.DropTimeFixed,   TrackParams.RevisitRateUB[1])
        TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[1])
    else:
        TrackParams.JobDuration = np.append( TrackParams.JobDuration ,  TrackParams.DwellTime[2]  )
        TrackParams.JobSlope = np.append(TrackParams.JobSlope, TrackParams.Slope[2])
        TrackParams.DropTime = np.append(TrackParams.DropTime, TrackParams.RevisitRateUB[2])
        TrackParams.DropTimeFixed = np.append( TrackParams.DropTimeFixed,   TrackParams.RevisitRateUB[2])
        TrackParams.DropCost = np.append(TrackParams.DropCost, TrackParams.Penalty[2])





#%% Begin Scheduler Loop




# import numpy as np
from radar_scheduling.tasks import ReluDrop

# rng = np.random.default_rng(100)
# task_gen = ReluDrop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
#                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
# tasks = task_gen.rand_tasks(N)

# A = list()
job = []
cnt = 0 # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
for ii in range(Nsearch):
    job.append(ReluDrop(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
    job[ii].Id = cnt # Numeric Identifier for each job
    cnt = cnt + 1
    if job[ii].slope == 0.4:
        job[ii].Type = 'HS' # Horizon Search (Used to determine revisit rates by job type
    else:
        job[ii].Type = 'AHS' # Above horizon search
    job[ii].Priority = job[ii](0) # Priority used to select which jobs to give to scheduler

    # tasks = ReluDrop(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
    # A.append(tasks)
    # del tasks
for ii in range(Ntrack):
    job.append(ReluDrop(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
    job[cnt].Id = cnt # Numeric Identifier for each job
    if job[cnt].slope == 0.25:
        job[cnt].Type = 'Tlow' # Low Priority Track
    elif job[cnt].slope == 0.5:
        job[cnt].Type = 'Tmed' # Medium Priority Track
    else:
        job[cnt].Type = 'Thigh' # High Priority Track
    job[cnt].Priority = job[cnt](0)
    cnt = cnt + 1



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

## Metrics
Job_Revisit_Count = np.zeros(len(job))
# Job_Revisit_Time = []*len(job)
# Job_Revisit_Time = []
# Job_Revisit_Time = list(range(len(job)))
Job_Revisit_Time = []
for ii in range(len(job)): # Create a list of empty lists
    Job_Revisit_Time.append([])


# metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
# JobRevistTime{ queue(n).Id }( metrics.JobRevistCount(queue(n).Id) )     = timeSec;


## Begin Main Loop
for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
    for i_run in range(n_run):  # Perform new algorithm runs: Should never be more than 1 in this code

        ChannelAvailableTime = np.zeros(K)
        for ii in np.arange(NumSteps): # Main Loop to evaluate schedulers
            timeSec = ii*RP # Current time

            if timeSec % RP*10 == 0:
                print('time =', timeSec)

            if np.min(ChannelAvailableTime) > timeSec:
                continue # Jump to next Resource Period

            # Reassess Track Priorities
            for jj in range(len(job)):
                job[jj].Priority = job[jj](timeSec)

            priority = np.array([task.Priority for task in job])
            priority_Idx = np.argsort(-1*priority) # Note: Multiple by -1 to reverse order or [::-1] reverses sort order to be descending.

            job_scheduler = [] # Jobs to be scheduled (Length N)
            for nn in range(N):
                job_scheduler.append(job[priority_Idx[nn]]) # Copy desired job

            unwanted = priority_Idx[0:N]
            for ele in sorted(unwanted, reverse=True):
                del job[ele]

            # for nn in range(N):
                # job.pop(priority_Idx[nn]) # Use pop to remove jobs. Can't do pop in 206 because priority_Idx will not be correct


            _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {1}', clear=True)
            try:
                plot_task_losses(job_scheduler, ax=ax_gen[0])
            except:
                print("Something went wrong")
            # else:
            #     print("Nothing went wrong")




            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex, T = alg_func(job_scheduler, ChannelAvailableTime) # Added Sequence T
            t_run = time.time() - t_start

            # Update ChannelAvailable Time
            duration = np.array([task.duration for task in job_scheduler])
            t_complete = t_ex + duration
            for kk in range(K):
                ChannelAvailableTime[kk] = np.max(t_complete[ch_ex == kk])


            check_valid(job_scheduler, t_ex, ch_ex)
            l_ex = eval_loss(job_scheduler, t_ex)

            t_run_iter[alg_repr][ii, i_run] = t_run
            l_ex_iter[alg_repr][ii, i_run] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

            t_run_mean[alg_repr][ii] = t_run_iter[alg_repr][ii].mean()
            l_ex_mean[alg_repr][ii] = l_ex_iter[alg_repr][ii].mean()

            if timeSec % RP * 10 == 0:
                print('')
                print(f"    Avg. Runtime: {t_run_mean[alg_repr][ii]:.2f} (s)")
                print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][ii]:.2f}")
                scatter_loss_runtime(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])

            # plot_results(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])
            # plot_loss_runtime(max_runtimes, l_ex_mean[i_gen], ax=ax_gen[1])
            for n in range(len(job_scheduler)):
                job_scheduler[n].t_release = t_ex[n] + job_scheduler[n].duration # Update Release Times based on execution + duration
                job_scheduler[n].t_drop = job_scheduler[n].t_release + job_scheduler[n].t_drop_fixed # Update Drop time from new start time
                job.append(job_scheduler[n])
                # print(job_scheduler[n].Id)
                Job_Revisit_Count[job_scheduler[n].Id] = Job_Revisit_Count[job_scheduler[n].Id] + 1
                Job_Revisit_Time[job_scheduler[n].Id].append(timeSec)
                # TODO: Update Drop Times - Done

                # new_job.append( job_scheduler.pop)
                # new_job(n).Id = queue((n)).Id;
            #     new_job(n).StartTime = t_ex((n)) + queue((n)).Duration;
            #     new_job(n).slope = queue((n)).slope;
            #     new_job(n).DropTime = queue((n)).DropTime;
            #     new_job(n).DropRelativeTime = queue((n)).DropTime + new_job(n).StartTime; % Update with new start time and job DropTime
            #     new_job(n).DropCost = queue((n)).DropCost;
            #     new_job(n).Duration = queue((n)).Duration;
            #     new_job(n).Type = queue((n)).Type;
                
                # metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
                # JobRevistTime{queue(n).Id}(metrics.JobRevistCount(queue(n).Id)) = timeSec;

            # TODO Put jobs in job_scheduler at the end of the master list "job", Finish plotting - Done

## Performance Assessment
# IMPORTANT: Job_Revisit_Time  records when the jobs are visited. We want the "Revisit Rate" which requires the np.diff below
mean_revisit_time = np.array([np.mean(np.diff(RT)) for RT in Job_Revisit_Time])


plt.figure(100)
plt.plot(mean_revisit_time)
plt.xlabel('Job ID')
plt.ylabel('Revisit Rate')
plt.show()
a = 1

# Display Revisit Rate by Job Type
job_type = np.array([task.Type for task in job])
job_unique = np.unique(job_type)
index_pos_list = []
for type in job_unique:
    job_type == type
    A = np.where(job_type == type)
    index_pos_list.append(A[:])

mean_revisit_time_job_type = np.zeros(len(job_unique))
for ii in range(len(job_unique)):
    mean_revisit_time_job_type[ii] = np.mean(   mean_revisit_time[index_pos_list[ii]]   )













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


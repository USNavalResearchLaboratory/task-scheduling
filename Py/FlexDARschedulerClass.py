def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step



import numpy as np
import matplotlib.pyplot as plt
import time     # TODO: use builtin module timeit instead? or cProfile?
import random
import math
from SL_policy_Discrete import load_policy, wrap_policy




random.seed(30)

#%% Set up Problem
N = 4 # Number of Jobs to process simultaneously
K = 1 # Number of timelines (radars)
RP = 0.04
# Tmax = 3

## Specify Algorithms
from tree_search import branch_bound, random_sequencer, earliest_release
from functools import partial
from util.generic import algorithm_repr, check_rng
from util.plot import plot_task_losses, plot_schedule, plot_loss_runtime, scatter_loss_runtime
from util.results import check_valid, eval_loss
from more_itertools import locate

from math import factorial, floor


# policy_file = 'temp/2020-08-03_11-08-06'
policy_file = 'temp/2020-08-03_11-08-06'
network_policy = load_policy(policy_file)

alg_funcs = [partial(earliest_release, do_swap=True),
             partial(random_sequencer),
             partial(network_policy),
             partial(branch_bound, verbose=False)]

             #partial(branch_bound, verbose=False),
             # partial(mc_tree_search, n_mc=[floor(.1 * factorial(n)) for n in range(n_tasks, 0, -1)], verbose=False),
             #
             # partial(random_agent)]
alg_n_runs = [1]*len(alg_funcs)       # number of runs per problem
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
SearchParams.Penalty = 300*np.ones(np.shape(SearchParams.RevistRate)) # Penalty for exceeding UB
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
from tasks import ReluDropGenerator
from tasks import ReluDropTask

# rng = np.random.default_rng(100)
# task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
#                              t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator
# tasks = task_gen.rand_tasks(N)

# A = list()
job = []
cnt = 0 # Make 0-based, saves a lot of trouble later when indexing into python zero-based vectors
for ii in range(Nsearch):
    # job.append(0, ReluDropTask(SearchParams.JobDuration[ii], SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropTimeFixed[ii], SearchParams.DropCost[ii]))
    job.append(ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii]))
    job[ii].Id = cnt # Numeric Identifier for each job
    cnt = cnt + 1
    if job[ii].slope == 0.4:
        job[ii].Type = 'HS' # Horizon Search (Used to determine revisit rates by job type
    else:
        job[ii].Type = 'AHS' # Above horizon search
    job[ii].Priority = job[ii](0) # Priority used to select which jobs to give to scheduler

    # tasks = ReluDropTask(SearchParams.JobDuration[ii], 0, SearchParams.JobSlope[ii], SearchParams.DropTime[ii], SearchParams.DropCost[ii])
    # A.append(tasks)
    # del tasks
for ii in range(Ntrack):
    # job.append(ReluDropTask(0, TrackParams.JobDuration[ii], TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropTimeFixed[ii], TrackParams.DropCost[ii]))
    job.append(ReluDropTask(TrackParams.JobDuration[ii], 0, TrackParams.JobSlope[ii], TrackParams.DropTime[ii], TrackParams.DropCost[ii]))
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
MaxTime = 20
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
Job_Revisit_Count = np.zeros((len(job), len(alg_reprs)))
# Job_Revisit_Time = []*len(job)
# Job_Revisit_Time = []
# Job_Revisit_Time = list(range(len(job)))
Job_Revisit_Time = []
for ii in range(len(job)): # Create a list of empty lists --> make it multi-dimensional to support different algorithms
    elem = []
    for jj in range(len(alg_reprs)):
        elem.append([])
    Job_Revisit_Time.append(elem)

RunTime = np.zeros((NumSteps,len(alg_reprs)))
Cost = np.zeros((NumSteps,len(alg_reprs)))

# metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
# JobRevistTime{ queue(n).Id }( metrics.JobRevistCount(queue(n).Id) )     = timeSec;


## Begin Main Loop
record = {'t_release': np.empty([NumSteps, len(alg_reprs), N]), 'duration': np.empty([NumSteps, len(alg_reprs), N]),
          'slope': np.empty([NumSteps, len(alg_reprs), N]), 'drop_time': np.empty([NumSteps, len(alg_reprs), N]),
          'drop_loss': np.empty([NumSteps, len(alg_reprs), N]), 'ch_avail': np.empty([NumSteps, len(alg_reprs), K])}
record['t_release'][:] = np.NaN
record['duration'][:] = np.NaN
record['slope'][:] = np.NaN
record['drop_time'][:] = np.NaN
record['drop_loss'][:] = np.NaN
record['ch_avail'][:] = np.NaN


# record = TaskParameters() # Initialize as empty class
# record.t_release = np.array([])
# record.duration = np.array([])
# record.slope = np.array([])
# record.drop_time = np.array([])
# record.drop_loss = np.array([])

job_type = np.array([task.Type for task in job]) # Original ordering of job types. Needed for metrics later
UB_job_type = np.array([1/task.slope for task in job])

idx_alg = 0
for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
    print(alg_repr)
    for ii in range(len(job)): # Reset Release Times to 0 for all jobs
        job[ii].t_release = 0
    for i_run in range(n_run):  # Perform new algorithm runs: Should never be more than 1 in this code

        ChannelAvailableTime = np.zeros(K)
        for ii in np.arange(NumSteps): # Main Loop to evaluate schedulers
            timeSec = ii*RP # Current time

            if np.min(ChannelAvailableTime) > timeSec:
                RunTime[ii, idx_alg] = math.nan
                Cost[ii, idx_alg] = math.nan
                continue # Jump to next Resource Period

            if timeSec % RP*1 == 0:
                print('time =', timeSec)

            # Reassess Track Priorities
            for jj in range(len(job)):
                job[jj].Priority = job[jj](timeSec)

            priority = np.array([task.Priority for task in job])
            priority_Idx = np.argsort(-1*priority, kind='mergesort') # Note: default 'quicksort' gives strange ordering Note: Multiple by -1 to reverse order or [::-1] reverses sort order to be descending.

            if False:
                task_ID = np.array([task.Id for task in job])
                release_time = np.array([task.t_release for task in job])
                task_sort = np.argsort(task_ID)
                plt.figure(10 + idx_alg)
                plt.clf()
                # time.sleep(0.3)
                plt.plot(task_ID[task_sort], priority[task_sort], marker='o', label='priority')
                plt.plot(task_ID[task_sort], release_time[task_sort], marker='d', label='release time')
                plt.title(alg_repr + ' Time = {:0.2f}'.format(timeSec))
                plt.show()


            job_scheduler = [] # Jobs to be scheduled (Length N)
            for nn in range(N):
                job_scheduler.append(job[priority_Idx[nn]]) # Copy desired job

            record['t_release'][ii, idx_alg, :] = np.array([task.t_release for task in job_scheduler])
            record['duration'][ii, idx_alg, :] = np.array([task.duration for task in job_scheduler])
            record['slope'][ii, idx_alg, :] = np.array([task.slope for task in job_scheduler])
            record['drop_time'][ii, idx_alg, :] = np.array([task.t_drop for task in job_scheduler])
            record['drop_loss'][ii, idx_alg, :] = np.array([task.l_drop for task in job_scheduler])
            record['ch_avail'][ii, idx_alg, :] = ChannelAvailableTime

            # RECORD[ii,idx_alg] = [task.t_release for task in job_scheduler]



            unwanted = priority_Idx[0:N]
            for ele in sorted(unwanted, reverse=True):
                del job[ele]

            # for nn in range(N):
                # job.pop(priority_Idx[nn]) # Use pop to remove jobs. Can't do pop in 206 because priority_Idx will not be correct


            # _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {1}', clear=True)
            # plot_task_losses(job_scheduler, ax=ax_gen[0])

            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            t_ex, ch_ex = alg_func(job_scheduler, ChannelAvailableTime) # Added Sequence T
            t_run = time.time() - t_start


            check_valid(job_scheduler, t_ex, ch_ex)
            l_ex = eval_loss(job_scheduler, t_ex)

            t_run_iter[alg_repr][ii, i_run] = t_run
            l_ex_iter[alg_repr][ii, i_run] = l_ex
            RunTime[ii,idx_alg] = t_run
            Cost[ii,idx_alg] = l_ex

            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, alg_repr=alg_repr, ax=None)

            t_run_mean[alg_repr][ii] = t_run_iter[alg_repr][ii].mean()
            l_ex_mean[alg_repr][ii] = l_ex_iter[alg_repr][ii].mean()

            if timeSec % RP * 10 == 0:
                print('')
                print(f"    Avg. Runtime: {t_run_mean[alg_repr][ii]:.2f} (s)")
                print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][ii]:.2f}")
                # scatter_loss_runtime(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])

            # Logic to put tasks that are scheduled after RP back on job stack
            # Update ChannelAvailable Time
            duration = np.array([task.duration for task in job_scheduler])
            t_complete = t_ex + duration
            executed_tasks = t_complete <= timeSec + RP # Task that are executed
            for kk in range(K):
                ChannelAvailableTime[kk] = np.max(t_complete[(ch_ex == kk) & executed_tasks])

            # plot_results(t_run_iter[ii], l_ex_iter[ii], ax=ax_gen[1])
            # plot_loss_runtime(max_runtimes, l_ex_mean[i_gen], ax=ax_gen[1])
            for n in range(len(job_scheduler)):
                if executed_tasks[n]: # Only updated executed tasks
                    job_scheduler[n].t_release = t_ex[n] + job_scheduler[n].duration # Update Release Times based on execution + duration
                    Job_Revisit_Count[job_scheduler[n].Id, idx_alg] = Job_Revisit_Count[job_scheduler[n].Id, idx_alg] + 1
                    Job_Revisit_Time[job_scheduler[n].Id][idx_alg].append(timeSec)
                # job_scheduler[n].t_drop = job_scheduler[n].t_release + job_scheduler[n].t_drop_fixed # Update Drop time from new start time   TODO: delete?
                job.append(job_scheduler[n])
                # print(job_scheduler[n].Id)

                # TODO: Update Drop Times - Done



    idx_alg = idx_alg + 1  # Increment which algorithm is being examined


                
                # metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
                # JobRevistTime{queue(n).Id}(metrics.JobRevistCount(queue(n).Id)) = timeSec;

            # TODO Put jobs in job_scheduler at the end of the master list "job", Finish plotting - Done
## Performance Assessment
A = np.subtract( record['t_release'] , np.max(record['ch_avail'],axis=2)[:,:,None] )
B = np.subtract( record['t_release'] , record['drop_time'] )

for ii in range(len(alg_reprs)):
    plt.figure(97+ii)
    ax1 = plt.subplot(321)
    plt.hist(np.ravel(record['t_release'][:, ii, :]), density=False, bins=100)
    plt.xlabel('t_release')
    plt.title(alg_reprs[ii])
    ax2 = plt.subplot(322)
    plt.hist(np.ravel(record['duration'][:, ii, :]), density=False, bins=100)
    plt.xlabel('duration')
    ax3 = plt.subplot(323)
    plt.hist(np.ravel(record['slope'][:, ii, :]), density=False, bins=100)
    plt.xlabel('slope')
    ax4 = plt.subplot(324)
    plt.hist(np.ravel(record['drop_time'][:, ii, :]), density=False, bins=100)
    plt.xlabel('drop_time')
    ax5 = plt.subplot(325)
    plt.hist(np.ravel(A[:, ii, :]), density=True, bins=100)
    plt.xlabel('t_release - max(ch_avail)')
    ax6 = plt.subplot(326)
    plt.hist(np.ravel(B[:, ii, :]), density=True, bins=100)
    plt.xlabel('t_release - drop_time')




Alg_time = np.nanmean(RunTime, axis = 0)
Alg_cost = np.nanmean(Cost, axis = 0)
plt.figure(99)
for ii in range(len(alg_reprs)):
    plt.plot(Alg_time[ii], Alg_cost[ii], marker = "o", label = alg_reprs[ii])
plt.xlabel('Run Time')
plt.ylabel('Cost')
plt.legend()
plt.show()


# scatter_loss_runtime(t_run_iter[0], l_ex_iter[0], ax=ax_gen[1])

# TODO: Create Utility Function --> how often algorithms go beyond bounds, verify everything is working correctly



# IMPORTANT: Job_Revisit_Time  records when the jobs are visited. We want the "Revisit Rate" which requires the np.diff below
# TODO: Redesign for multiple algorithms
mean_revisit_time = np.zeros((len(job), len(alg_reprs)))
for ii in range(len(job)):
    for jj in range(len(alg_reprs)):
        # mean_revisit_time[ii,jj] = np.mean(np.diff(np.append(0,Job_Revisit_Time[ii][jj]))) # Add 0 for cases where there is only 1 visit
        if len(Job_Revisit_Time[ii][jj]) > 1:
            mean_revisit_time[ii,jj] = np.mean(np.diff(Job_Revisit_Time[ii][jj])) # Add 0 for cases where there is only 1 visit
        else:
            mean_revisit_time[ii,jj] = np.nan
# mean_revisit_time = np.array([np.mean(np.diff(RT)) for RT in Job_Revisit_Time])

# Display Revisit Rate by Job Type
# job_type = np.array([task.Type for task in job])
job_unique = np.unique(job_type)
N_job_types = len(job_unique)
index_pos_list = []
for type in job_unique:
    job_type == type
    A = np.where(job_type == type)
    index_pos_list.append(A[:])

# mean_revisit_time_job_type = np.zeros(len(job_unique))
# for ii in range(len(job_unique)):
#     mean_revisit_time_job_type[ii] = np.mean(   mean_revisit_time[index_pos_list[ii]]   )


idx_sort = np.argsort(job_type)
last_index = np.zeros(len(job_unique))
for jj in range(len(job_unique)):
    # last_index[jj] = np.where(job_type == job_unique[jj])[0][-1]
    last_index[jj] = np.where(job_type[idx_sort] == job_unique[jj])[0][-1]

last_index = last_index.astype(int)
first_index = np.append(0, last_index[0:N_job_types-1]+1 ).astype(int)
UB_revisit_rate = np.array(np.zeros(len(first_index)))
for jj in range(len(first_index)):
    UB_revisit_rate[jj] = UB_job_type[idx_sort[first_index[jj]]]
# UB_revisit_rate = np.array([job[idx_sort[first_index[idx]]].t_drop for idx in range(N_job_types)])
# desired_revisit_rate = job[idx_sort[first_index]].t_drop

mean_revisit_time_job_type = np.zeros((len(job_unique),len(alg_reprs)))
Utility = np.zeros((len(job_unique),len(alg_reprs)))
Penalty = np.zeros((len(job_unique),len(alg_reprs)))
for ii in range(len(job_unique)):
    for jj in range(len(alg_reprs)):
        idx_support = idx_sort[first_index[ii]:last_index[ii]]
        mean_revisit_time_job_type[ii,jj] = np.mean(mean_revisit_time[idx_support,jj])
        temp = UB_revisit_rate[ii] - mean_revisit_time[idx_support,jj]
        Utility[ii,jj] = np.sum(temp)
        Penalty[ii,jj] = -np.sum(temp[temp<0])




color_scheme_bound = ['b', 'g', 'r', 'm', 'k']
color_scheme = [['crimson', 'green', 'navy', 'magenta', 'darkred'], ['cyan', 'teal','navy','aquamarine'], ['lime', 'yellowgreen', 'chartreuse', 'lightgreen'],
                ['magenta', 'maroon', 'voilet', 'fushia'], ['grey', 'chocolate', 'brown', 'beige'] ]

for jj in range(len(alg_reprs)):
    plt.figure(100+jj)
    plt.grid
    for ii in range(N_job_types):
        y = np.arange(first_index[ii],last_index[ii]+1)
        x = UB_revisit_rate[ii] * np.ones(np.shape(y))
        if ii == 0:
            plt.plot(x, y, color_scheme_bound[0], label='UB')
        else:
            plt.plot(x, y, color_scheme_bound[0])
        plt.text(x[0], y[0], 'Upper-Bound: '+job_unique[ii])
        if ii == 0:
            plt.plot(mean_revisit_time[idx_sort[y], jj], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])
        else:
            plt.plot(mean_revisit_time[idx_sort[y], jj], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])

        # y2 = mean_revisit_time_job_type[ii,jj] * np.ones(np.shape(x))
        # plt.plot(x, y2, color_scheme[ii])
        # plt.text(x[0], y2[0], 'Mean: '+job_unique[ii])
    plt.ylabel('Sorted Job ID')
    plt.xlabel('Revisit Rate')
    plt.grid(True)
    plt.show()
    plt.legend()
    plt.title(alg_reprs[jj] + '\n Penalty = ' + str(np.sum(Penalty[:,jj])) )


plt.figure(200)
plt.plot(job_unique, mean_revisit_time_job_type, label=alg_reprs[jj])
plt.xlabel('Job Type')
plt.ylabel('Mean Revisit Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(201)
for jj in range(len(alg_reprs)):
    plt.plot(job_unique, Penalty[:,jj], label=alg_reprs[jj])
plt.ylabel('Penalty')
plt.xlabel('Job Type')
plt.grid(True)
plt.legend()
plt.show()

Alg_Penalty = np.sum(Penalty,axis=0)
plt.figure(202)
for jj in range(len(alg_reprs)):
    plt.plot(Alg_time[jj], Alg_Penalty[jj], marker='o', label=alg_reprs[jj])
plt.xlabel('Run Time (s)')
plt.ylabel('Penalty')
plt.legend()





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


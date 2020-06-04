
import numpy as np

# Set up Problem
N = 8
RP = 0.04
Tmax = 50


class TaskParameters: # Initializes to something like matlab structure. Enables dot indexing
    pass

# Generate Search Tasks
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

# Generate Track Tasks
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








import numpy as np
from tasks import ReluDropGenerator
rng = np.random.default_rng(100)

task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                             t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator

tasks = task_gen.rand_tasks(N)


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


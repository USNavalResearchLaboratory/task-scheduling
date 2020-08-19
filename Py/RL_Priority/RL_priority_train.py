
import numpy as np
import pickle
import kneed
import os
import sys, os
import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../anotherproject")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from RL_Priority.env_priority import PriorityQueue

Queue = PriorityQueue()

# Reassess Track Priorities
job = Queue.job
timeSec = Queue.timeSec
for jj in range(len(job)):
    job[jj].Priority = job[jj](timeSec)

priority = np.array([task.Priority for task in job])
priority_Idx = np.argsort(-1*priority, kind='mergesort') # Note: default 'quicksort' gives strange ordering Note: Multiple by -1 to reverse order or [::-1] reverses sort order to be descending.



# from kneed import KneeLocator


# Load Pickled Data
print(os.getcwd())
filename = './RL_data/RLP_data_2020-08-06_09-53-56'
infile = open(filename, 'rb')
data_dict = pickle.load(infile)
infile.close()

a = 1

F = data_dict.get('F')
feat_record = data_dict.get('feat_record')
feature_names = data_dict.get('feature_names')
a = 1

for jj in range(len(feat_record)):
    plt.figure(jj)
    plt.style.use("fivethirtyeight")
    plt.hist(F[:,jj])
    plt.title(feature_names[jj])





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

NumSteps = Queue.NumSteps
NumEpoch = 10
reward = np.empty([NumEpoch, NumSteps])
for i_episode in range(NumEpoch):
    observation = Queue.reset()
    timeSec = Queue.timeSec
    RP = Queue.env_config.get("RP")

    for ii in np.arange(0, NumSteps):  # Main Loop to evaluate schedulers (Start at 1 because PriorityQueue() Initializes time 0
        timeSec = ii * RP  # Current time

        ChannelAvailableTime = Queue.env_config.get("ChannelAvailableTime")
        Queue.timeSec = timeSec

        if np.min(ChannelAvailableTime) > timeSec:
            # RunTime[ii, idx_alg] = math.nan
            # Cost[ii, idx_alg] = math.nan
            continue  # Jump to next Resource Period



        action = Queue.action_space.sample()  # Insert your policy here

        N = Queue.env_config.get("N")
        action = np.ones(N, dtype='int64')  # Always choose first action available

        ob_next, reward[i_episode, ii], done, info = Queue.step(action)

    # for t in range(100):
    #     # env.render()
    #     print(observation)
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         print("Episode finished after {} timesteps".format(t+1))
    #         break
    # env.close()






# Plot Rewards
plt.figure(1)
plt.plot(np.mean(reward, axis=0))

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




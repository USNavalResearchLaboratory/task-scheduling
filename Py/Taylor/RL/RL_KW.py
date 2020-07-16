

import numpy as np
import matplotlib.pyplot as plt



N = 3
MC = 1000

max_slope = 10
max_release = 1
max_duration = 1

t_release = np.random.uniform(0, max_release, (MC, N))
slope = np.random.uniform(0, max_slope, (MC, N))
duration = np.random.uniform(0, max_duration, (MC, N))

# Discretize Our Variables
num_steps = 10 # Number of discrete values in each variable. For now just make them all the same. Can change in the future
dt = max_release/num_steps
ds = max_slope/num_steps
dd = max_duration/num_steps

# t_release*


data = np.concatenate((t_release, slope, duration), axis=1)


## INITIALIZE Q Learning

num_actions = N



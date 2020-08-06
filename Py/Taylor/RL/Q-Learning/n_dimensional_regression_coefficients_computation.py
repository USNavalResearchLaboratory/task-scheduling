#%% Terminal:

import os
os.system('cls')

print("\n")


#%% Import Statements:

from task_generator_with_EST import task_generator_with_EST_function
from observations_to_state_release_times import observations_to_state_release_times_function
from observations_to_state_weights import observations_to_state_weights_function
from observations_to_state_lengths import observations_to_state_lengths_function
from observations_to_state_q_matrix import observations_to_state_q_matrix_function
from n_dimensional_regression import n_dimensional_regression_function

import numpy


#%% User-Specified Inputs:

number_of_sets = 1000000

number_of_tasks = 3

number_of_features = 3

maximum_weight_of_tasks = 10

order_of_features = {'Release Times': 1,'Weights': 2,'Lengths': 3}

steps_for_feature_discretization = {'Release Times': 3,'Weights': 3,'Lengths': 3}


#%% Definitions of Variables Based on User-Specified Inputs:

r_N_index = order_of_features["Release Times"]
w_N_index = order_of_features["Weights"]
l_N_index = order_of_features["Lengths"]

features_indices = [r_N_index,w_N_index,l_N_index]

r_N_steps = steps_for_feature_discretization["Release Times"]
w_N_steps = steps_for_feature_discretization["Weights"]
l_N_steps = steps_for_feature_discretization["Lengths"]

number_of_steps = [r_N_steps,w_N_steps,l_N_steps]


#%% Task Generator with Earliest Start Time (EST) Algorithm:

inputs = task_generator_with_EST_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks,r_N_index)

F = inputs[:,:number_of_tasks*number_of_features]

r_N = F[:,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]
w_N = F[:,(w_N_index-1)*number_of_tasks:(w_N_index*number_of_tasks)]
l_N = F[:,(l_N_index-1)*number_of_tasks:(l_N_index*number_of_tasks)]


#%% Coefficients for N-Dimensional Regression:

r_states = numpy.zeros(len(r_N))
w_states = numpy.zeros(len(w_N))
l_states = numpy.zeros(len(l_N))

q_states = numpy.zeros(number_of_sets)

X = numpy.ones([number_of_sets,number_of_features+1])
Y = numpy.ones([number_of_sets,1])

for i in range(number_of_sets):

    r_states[i] = observations_to_state_release_times_function(number_of_tasks,r_N[i,:],r_N_steps)
    w_states[i] = observations_to_state_weights_function(number_of_tasks,w_N[i,:],w_N_steps)
    l_states[i] = observations_to_state_lengths_function(number_of_tasks,l_N[i,:],l_N_steps)

    q_states[i] = observations_to_state_q_matrix_function(number_of_tasks,r_N_steps,w_N_steps,l_N_steps,r_states[i],w_states[i],l_states[i])

    X[i,1] = r_states[i]
    X[i,2] = w_states[i]
    X[i,3] = l_states[i]

    Y[i] = q_states[i]

B = n_dimensional_regression_function(X,Y)

print("\nCoefficients = \n\n{}".format(B))


#%% Terminal:

print("\n")


#%% Documentation:

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html


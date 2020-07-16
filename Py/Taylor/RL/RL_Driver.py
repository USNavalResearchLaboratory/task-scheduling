import os
os.system('cls')


##### Formatting #####

print("\n")


##### Import Statements #####

from task_generation_with_EST_algorithm import task_generation_with_EST_algorithm_function
from cost import cost_function
from training import training_function
from testing import testing_function
from analysis_EST import analysis_EST_function

import itertools
import math
import numpy


##### User-Specified Inputs (Environment) #####

number_of_sets_training = 10000 # !!!!! HERE !!!!!
number_of_sets_testing = 100 # !!!!! HERE !!!!!

number_of_tasks = 3

number_of_features = 3

maximum_weight_of_tasks = 10


##### Environment (Observations) #####

training_inputs = task_generation_with_EST_algorithm_function(number_of_sets_training,number_of_tasks,number_of_features,maximum_weight_of_tasks)
testing_inputs = task_generation_with_EST_algorithm_function(number_of_sets_testing,number_of_tasks,number_of_features,maximum_weight_of_tasks)

F_training = training_inputs[:,0:number_of_tasks*number_of_features]
F_testing = testing_inputs[:,0:number_of_tasks*number_of_features]

EST_training = training_inputs[:,number_of_tasks*number_of_features:]
EST_testing = testing_inputs[:,number_of_tasks*number_of_features:]

# print("\nF (Training) = \n\n{}".format(F_training))
# print("\nF (Testing) = \n\n{}".format(F_testing))

# print("\nEST (Training) = \n\n{}".format(EST_training))
# print("\nEST (Testing) = \n\n{}".format(EST_testing))

r_N_training = F_training[:,0:1*number_of_tasks]
w_N_training = F_training[:,1*number_of_tasks:2*number_of_tasks]
l_N_training = F_training[:,2*number_of_tasks:3*number_of_tasks]

r_N_testing = F_testing[:,0:1*number_of_tasks]
w_N_testing = F_testing[:,1*number_of_tasks:2*number_of_tasks]
l_N_testing = F_testing[:,2*number_of_tasks:3*number_of_tasks]


##### Cost / Reward #####

print("\n\nExecuting 'cost.py' ...")

cost_matrix = [numpy.zeros([number_of_sets_training,math.factorial(number_of_tasks)])][0]

permutations = itertools.permutations(list(range(1,number_of_tasks+1)))

count = 0

for i in list(permutations):

    sequence = numpy.asarray(i)

    sequences = numpy.zeros([number_of_sets_training,number_of_tasks])

    for j in range(number_of_sets_training):

        sequences[j,:] = sequence

    cost = cost_function(r_N_training,w_N_training,l_N_training,sequences)

    cost_matrix[:,count] = cost[:,0]

    count += 1

# print("\nCost Matrix = \n\n{}".format(cost_matrix))
print("\n\tCost Matrix = {}".format(numpy.shape(cost_matrix)))

reward_matrix = numpy.zeros([numpy.shape(cost_matrix)[0],numpy.shape(cost_matrix)[1]])

for i in range(numpy.shape(cost_matrix)[0]):

    for j in range(numpy.shape(cost_matrix)[1]):

        cost_value = cost_matrix[i,j]

        if cost_value == 0:
            reward_value = 100 # !!!!! HERE !!!!!
        else:
            reward_value = 1/cost_value
            
        reward_matrix[i,j] = reward_value

# print("\nReward Matrix = \n\n{}".format(reward_matrix))
print("\n\tReward Matrix = {}".format(numpy.shape(reward_matrix)))


##### Training (Q-Learning) #####

number_of_states = [3,3,3] # !!!!! HERE !!!!!

episodes = 10 # !!!!! HERE !!!!!

alpha = 0.75 # !!!!! HERE !!!!!
gamma = 0.00 # !!!!! HERE !!!!!
epsilon = 0.75 # !!!!! HERE !!!!!

Q = training_function(number_of_tasks,number_of_states,F_training,reward_matrix,episodes,alpha,gamma,epsilon)


##### Testing #####

results = testing_function(number_of_tasks,F_testing,Q,number_of_states)


##### Analysis (EST and Total Loss => Cost) #####

permutations_for_analysis = itertools.permutations(list(range(1,number_of_tasks+1)))

permutations_for_analysis_as_arrays = numpy.zeros([math.factorial(number_of_tasks),number_of_tasks])

count = 0

for i in list(permutations_for_analysis):

    sequence = numpy.asarray(i)

    permutations_for_analysis_as_arrays[count,:] = sequence-1

    count += 1

actions_EST = numpy.zeros(number_of_sets_testing)

for i in range(len(EST_testing)):
    
    an_array = EST_testing[i,:]

    for j in range(len(permutations_for_analysis_as_arrays)):
        
        another_array = permutations_for_analysis_as_arrays[j,:]
        
        comparison = an_array == another_array
        
        equal_arrays = comparison.all()

        if equal_arrays == True:

            actions_EST[i] = j

accuracy_EST = analysis_EST_function(actions_EST,results)

loss_matrix = numpy.zeros([len(results)])

for i in range(len(results)):
    
    action = results[i]

    loss_value = cost_matrix[i,action.astype('int')]

    loss_matrix[i] = loss_value

total_loss = numpy.sum(loss_matrix)

# print("\nLoss = \n\n{}".format(loss_matrix))
print("\n\tTotal Loss = {:0.1f}".format(total_loss))


##### Formatting #####

print("\n")


##### Documentation #####

# https://docs.python.org/2/library/itertools.html

# https://docs.python.org/3/library/math.html

# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
# https://numpy.org/doc/1.18/reference/generated/numpy.shape.html
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html

# Examples => https://amunategui.github.io/reinforcement-learning/index.html
#             https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
#             https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa#:~:text=Many%20reinforcement%20learning%20introduce%20the,agent%20starting%20from%20state%20s%20.&text=Q%20is%20a%20function%20of,and%20returns%20a%20real%20value


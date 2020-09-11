#%% Terminal:

import os, sys
os.system('cls')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


print("\n")


#%% Import Statements:

from task_generator_with_EST import task_generator_with_EST_function
from training import training_function
from training_with_n_dimensional_regression import training_with_n_dimensional_regression_function
from training_with_equation import training_with_equation_function
from testing import testing_function
from testing_with_n_dimensional_regression import testing_with_n_dimensional_regression_function
from testing_with_equation import testing_with_equation_function
from analysis_EST import analysis_EST_function
from cost import cost_function

import numpy
import itertools
import math
import matplotlib.pyplot


#%% User-Specified Inputs:

number_of_sets_training = 100000
number_of_sets_testing = 100

number_of_tasks = 3

number_of_features = 3

maximum_weight_of_tasks = 10

order_of_features = {'Release Times': 1,'Weights': 2,'Lengths': 3}

zero_cost_reward = 100

steps_for_feature_discretization = {'Release Times': 3,'Weights': 3,'Lengths': 3}

episodes = 30

alpha = 1.0
gamma = 0.00
epsilon = 0.10


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

training_inputs = task_generator_with_EST_function(number_of_sets_training,number_of_tasks,number_of_features,maximum_weight_of_tasks,r_N_index)
testing_inputs = task_generator_with_EST_function(number_of_sets_testing,number_of_tasks,number_of_features,maximum_weight_of_tasks,r_N_index)

F_training = training_inputs[:,:number_of_tasks*number_of_features]
F_testing = testing_inputs[:,:number_of_tasks*number_of_features]

EST_training = training_inputs[:,number_of_tasks*number_of_features:]
EST_testing = testing_inputs[:,number_of_tasks*number_of_features:]


#%% Permutations:

permutations = itertools.permutations(list(range(1,number_of_tasks+1)))

sequences = numpy.zeros([math.factorial(number_of_tasks),number_of_tasks])

count = 0

for i in list(permutations):

    sequence = numpy.asarray(i)

    sequences[count,:] = sequence

    count += 1


#%% Training (Q-Learning):

# Q, total_loss_training_over_training, total_reward_training_over_training, total_loss_testing_over_training, total_reward_testing_over_training = training_function(number_of_tasks,features_indices,number_of_steps,F_training,F_testing,sequences,zero_cost_reward,episodes,alpha,gamma,epsilon)
# Q, total_loss_training_over_training, total_reward_training_over_training, total_loss_testing_over_training, total_reward_testing_over_training = training_with_n_dimensional_regression_function(number_of_tasks,features_indices,number_of_steps,F_training,F_testing,sequences,zero_cost_reward,episodes,alpha,gamma,epsilon)
Q, total_loss_training_over_training, total_reward_training_over_training, total_loss_testing_over_training, total_reward_testing_over_training = training_with_equation_function(number_of_tasks,features_indices,number_of_steps,F_training,F_testing,sequences,zero_cost_reward,episodes,alpha,gamma,epsilon)

#%% Testing:

print("\n\nExecuting 'testing.py' ...")

# results = testing_function(number_of_tasks,features_indices,number_of_steps,F_testing,Q)
# results = testing_with_n_dimensional_regression_function(number_of_tasks,features_indices,number_of_steps,F_testing,Q)
results = testing_with_equation_function(number_of_tasks,features_indices,number_of_steps,F_testing,Q)

print("\n\tTESTING FINISHED\n")


#%% Analysis (Comparison to EST):

results_EST = numpy.zeros(number_of_sets_testing)

for i in range(len(EST_testing)):

    an_array = EST_testing[i,:]

    for j in range(len(sequences)):

        another_array = sequences[j,:]-1

        comparison = an_array == another_array

        equal_arrays = comparison.all()

        if equal_arrays == True:

            results_EST[i] = j

comparison_to_EST = analysis_EST_function(results_EST,results)


#%% Analysis (Total Loss):

loss_after_training = numpy.zeros([len(results)])
loss_after_training_EST = numpy.zeros([len(results_EST)])

for i in range(len(results)):

    r_continuous = F_testing[i,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]
    w_continuous = F_testing[i,(w_N_index-1)*number_of_tasks:(w_N_index*number_of_tasks)]
    l_continuous = F_testing[i,(l_N_index-1)*number_of_tasks:(l_N_index*number_of_tasks)]

    action = results[i].astype('int')
    action_EST = results_EST[i].astype('int')

    sequence = sequences[action,:]
    sequence_EST = sequences[action_EST,:]

    r_continuous_reshape = numpy.reshape(r_continuous,(1,number_of_tasks))
    w_continuous_reshape = numpy.reshape(w_continuous,(1,number_of_tasks))
    l_continuous_reshape = numpy.reshape(l_continuous,(1,number_of_tasks))

    sequence_reshape = numpy.reshape(sequence,(1,number_of_tasks))
    sequence_EST_reshape = numpy.reshape(sequence_EST,(1,number_of_tasks))

    loss_value = cost_function(r_continuous_reshape,w_continuous_reshape,l_continuous_reshape,sequence_reshape)
    loss_value_EST = cost_function(r_continuous_reshape,w_continuous_reshape,l_continuous_reshape,sequence_EST_reshape)

    loss_after_training[i] = loss_value
    loss_after_training_EST[i] = loss_value_EST

total_loss_after_training = numpy.sum(loss_after_training)
total_loss_after_training_EST = numpy.sum(loss_after_training_EST)

print("\nTotal Loss After Training = {:0.1f}".format(total_loss_after_training))
print("\nTotal Loss After Training (EST) = {:0.1f}".format(total_loss_after_training_EST))


#%% Plots:

figure, (plot_1,plot_2,plot_3,plot_4) = matplotlib.pyplot.subplots(1,4)
figure.suptitle('Total Loss and Reward Over Training')
plot_1.plot(total_loss_training_over_training,'k')
plot_1.set_title('Total Loss (Training)')
plot_2.plot(total_reward_training_over_training,'b')
plot_2.set_title('Total Reward (Training)')
plot_3.plot(total_loss_testing_over_training,'k')
plot_3.set_title('Total Loss (Testing)')
plot_4.plot(total_reward_testing_over_training,'b')
plot_4.set_title('Total Reward (Testing)')


#%% Terminal:

print("\n")

matplotlib.pyplot.show()


#%% Documentation:

# https://docs.python.org/3/library/itertools.html

# https://docs.python.org/3/library/math.html

# https://matplotlib.org/api/pyplot_api.html

# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# https://numpy.org/doc/stable/reference/generated/numpy.sum.html

# Examples => https://amunategui.github.io/reinforcement-learning/index.html
#             https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
#             https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa#:~:text=Many%20reinforcement%20learning%20introduce%20the,agent%20starting%20from%20state%20s%20.&text=Q%20is%20a%20function%20of,and%20returns%20a%20real%20value
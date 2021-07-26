#%% Import Statements:

import numpy

from DQN_alternative_function import DQN_alternative_function


#%% User-Specified Inputs (Environment):

number_of_tasks = 3 # HEREHEREHERE!!!

number_of_features = 3

zero_cost_reward = 100 # HEREHEREHERE!!!

order_of_features = {'Release Times': 1,'Weights': 2,'Lengths': 3}


#%% User-Specified Inputs (Training):

epsilon = 0.99 # HEREHEREHERE!!!
epsilon_decay = 0.99 # HEREHEREHERE!!!
epsilon_minimum = 0.01 # HEREHEREHERE!!!

gamma = 0.99 # HEREHEREHERE!!!
gamma_decay = 0.99 # HEREHEREHERE!!!
gamma_minimum = 0.01 # HEREHEREHERE!!!

number_of_episodes = 50000 # MAXIMUM NUMBER OF SEQUENCES
replay_memory_size = 5000 # MAXIMUM NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)
minimum_replay_memory_size = 2500 # MINIMUM NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)
minibatch_size = 500 # ACTUAL NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)

update_target_every = 100 # NUMBER OF PAST SEQUENCES TO BE USED FOR UPDATING (TARGET MODEL)

batch_size = 100 # NUMBER OF PAST SEQUENCES TO BE USED FOR AVERAGING REWARD

number_of_simulations = 1 # HEREHEREHERE!!!

DQN_training_cost_average = numpy.zeros(number_of_simulations)
EST_training_cost_average = numpy.zeros(number_of_simulations)
DQN_testing_cost_average = numpy.zeros(number_of_simulations)
EST_testing_cost_average = numpy.zeros(number_of_simulations)

for i in range(1,number_of_simulations+1):

    DQN_training_cost_average[i], EST_training_cost_average[i], DQN_testing_cost_average[i], EST_testing_cost_average[i] = DQN_alternative_function(number_of_tasks,number_of_features,zero_cost_reward,order_of_features,
                                                                                                                                                    epsilon,epsilon_decay,epsilon_minimum,gamma,gamma_decay,gamma_minimum,
                                                                                                                                                    number_of_episodes,replay_memory_size,minimum_replay_memory_size,minibatch_size,update_target_every,batch_size)


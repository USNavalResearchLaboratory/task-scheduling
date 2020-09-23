#%% Import Statements:

import collections
import itertools
import math
import matplotlib.pyplot
import numpy
import random
import tensorflow

from cost import cost_function
from reward import reward_function
from task_generator_Integrated import task_generator_function
from tensorflow.python import keras


#%% User-Specified Inputs (Environment):

number_of_tasks = 3

number_of_features = 3

zero_cost_reward = 100 # HEREHEREHERE!!!

order_of_features = {'Release Times': 1,'Weights': 2,'Lengths': 3}


#%% User-Specified Inputs (Training):

epsilon = 1 # HEREHEREHERE!!!
epsilon_decay = 0.99 # HEREHEREHERE!!!
epsilon_minimum = 0.001 # HEREHEREHERE!!!

gamma = 1 # HEREHEREHERE!!!
gamma_decay = 0.99 # HEREHEREHERE!!!
gamma_minimum = 0.001 # HEREHEREHERE!!!

number_of_episodes = 10000 # MAXIMUM NUMBER OF SEQUENCES
replay_memory_size = int(number_of_episodes/10) # MAXIMUM NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)
minimum_replay_memory_size = int(replay_memory_size/10) # MINIMUM NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)
minibatch_size = int(minimum_replay_memory_size/10) # ACTUAL NUMBER OF PAST STEPS TO BE USED FOR TRAINING (MAIN MODEL)

update_target_every = 10 # NUMBER OF PAST SEQUENCES TO BE USED FOR UPDATING (TARGET MODEL)

batch_size = 100 # NUMBER OF PAST SEQUENCES TO BE USED FOR AVERAGING REWARD


#%% Definitions of Variables Based on User-Specified Inputs:

r_N_index = order_of_features["Release Times"]
w_N_index = order_of_features["Weights"]
l_N_index = order_of_features["Lengths"]


#%% Permutations:

permutations = itertools.permutations(list(range(1,number_of_tasks+1)))

sequences = numpy.zeros([math.factorial(number_of_tasks),number_of_tasks])

count = 0

for i in list(permutations):

    sequence = numpy.asarray(i)

    sequences[count,:] = sequence

    count += 1


#%% Environment:

class SchedulerEnv:

    observation_space_size = (1,number_of_tasks*number_of_features)
    action_space_size = math.factorial(number_of_tasks)

    def reset(self):

        observation = task_generator_function(number_of_tasks)

        return observation

    def step(self,observation,action):

        observation_array = numpy.array(observation)

        r_N = observation_array[(r_N_index-1),:]
        w_N = observation_array[(w_N_index-1),:]
        l_N = observation_array[(l_N_index-1),:]

        r_N_reshape = numpy.reshape(r_N,(1,number_of_tasks))
        w_N_reshape = numpy.reshape(w_N,(1,number_of_tasks))
        l_N_reshape = numpy.reshape(l_N,(1,number_of_tasks))

        sequence = sequences[action,:]

        sequence_reshape = numpy.reshape(sequence,(1,number_of_tasks))

        cost = cost_function(r_N_reshape,w_N_reshape,l_N_reshape,sequence_reshape)

        reward = reward_function(cost,zero_cost_reward)

        new_observation = task_generator_function(number_of_tasks)

        done = True # EVERY TASK IS SCHEDULED AT ONCE

        return reward, new_observation, done

env = SchedulerEnv()


#%% DQN:

class DQN:

    def __init__(self):

        #%% Main Model (Fit):
        self.model = self.create_model()

        #%% Target Model (Predict):
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #%% Replay Memory:
        self.replay_memory = collections.deque(maxlen=replay_memory_size)

        # Counter to Update Target Model:
        self.target_update_counter = 0

    def create_model(self): # HEREHEREHERE!!!

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(number_of_tasks*number_of_features,input_shape=(number_of_tasks*number_of_features,)))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(env.action_space_size,activation='linear'))

        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001),loss='mse',metrics=['accuracy'])

        return model

    def update_replay_memory(self,transition):

        self.replay_memory.append(transition)

    def train(self,terminal_state,step):

        if len(self.replay_memory) < minimum_replay_memory_size:

            return

        minibatch_before = random.sample(self.replay_memory,minibatch_size)

        minibatch_before_array = numpy.array(minibatch_before)

        minibatch_after_current_states = []
        minibatch_after_new_current_states = []

        for i in range(len(minibatch_before_array)):

            current_row = minibatch_before_array[i]

            current_state_in_current_row = current_row[0]
            new_current_state_in_current_row = current_row[3]

            r_N_in_current_state_in_current_row = current_state_in_current_row[r_N_index-1]
            w_N_in_current_state_in_current_row = current_state_in_current_row[w_N_index-1]
            l_N_in_current_state_in_current_row = current_state_in_current_row[l_N_index-1]

            r_N_in_new_current_state_in_current_row = new_current_state_in_current_row[r_N_index-1]
            w_N_in_new_current_state_in_current_row = new_current_state_in_current_row[w_N_index-1]
            l_N_in_new_current_state_in_current_row = new_current_state_in_current_row[l_N_index-1]

            current_state_row_to_append = numpy.concatenate((r_N_in_current_state_in_current_row,w_N_in_current_state_in_current_row,l_N_in_current_state_in_current_row),axis=0)
            new_current_state_row_to_append = numpy.concatenate((r_N_in_new_current_state_in_current_row,w_N_in_new_current_state_in_current_row,l_N_in_new_current_state_in_current_row),axis=0)

            minibatch_after_current_states.append(current_state_row_to_append)
            minibatch_after_new_current_states.append(new_current_state_row_to_append)

        current_states = numpy.array(minibatch_after_current_states)
        new_current_states = numpy.array(minibatch_after_new_current_states)

        current_q_values_all = self.model.predict(current_states)
        future_q_values_all = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state,action,reward,_,done) in enumerate(minibatch_before):

            r_N_in_current_state = current_state[r_N_index-1]
            w_N_in_current_state = current_state[w_N_index-1]
            l_N_in_current_state = current_state[l_N_index-1]

            current_state_to_append = numpy.concatenate((r_N_in_current_state,w_N_in_current_state,l_N_in_current_state),axis=0)

            if not done:

                max_future_q_value = numpy.max(future_q_values_all[index])

                new_q_value = reward+gamma*max_future_q_value

            else:

                new_q_value = reward

            current_q_values = current_q_values_all[index]
            current_q_values[action] = new_q_value

            X.append(current_state_to_append)
            Y.append(current_q_values)

        self.model.fit(numpy.array(X),numpy.array(Y),batch_size=minibatch_size,verbose=0,shuffle=False)

        if terminal_state:

            self.target_update_counter += 1

        if self.target_update_counter > update_target_every:

            self.target_model.set_weights(self.model.get_weights())

            self.target_update_counter = 0

    def get_q_values(self,state):

        return self.model.predict(numpy.reshape(numpy.array(state),env.observation_space_size))

agent = DQN()


#%% Training (DQN):

episode_reward_all = []

for episode in range(1,number_of_episodes+1):

    if episode % 1000 == 0:

        print(f"\n\tEpisode: {episode}")

    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False

    while not done:

        if numpy.random.random() > epsilon:

            action = numpy.argmax(agent.get_q_values(current_state))

        else:

            action = numpy.random.randint(0,env.action_space_size)

        reward, new_state, done = env.step(current_state,action)

        episode_reward += reward

        transition = (current_state,action,reward,new_state,done)

        agent.update_replay_memory(transition)

        agent.train(done,step)

        current_state = new_state

        step += 1

    episode_reward_all.append(episode_reward)

    number_of_batches = int(number_of_episodes/batch_size)

    episode_reward_average = numpy.zeros(number_of_batches)

    for i in range(number_of_batches):

        episode_reward_all_batch = episode_reward_all[i*batch_size:(i+1)*batch_size]
        episode_reward_average[i] = numpy.average(episode_reward_all_batch)

    if epsilon > epsilon_minimum:

        epsilon *= epsilon_decay

        epsilon = max(epsilon_minimum,epsilon)

    if gamma > gamma_minimum:

        gamma *= gamma_decay

        gamma = max(gamma_minimum,gamma)


#%% Plots:

figure, (plot_1,plot_2) = matplotlib.pyplot.subplots(1,2)
figure.suptitle('Training')
plot_1.plot(episode_reward_all,'k')
plot_1.set_title('Total Reward')
plot_2.plot(episode_reward_average,'k')
plot_2.set_title('Average Reward')

matplotlib.pyplot.show()


#%% Documentation:

# Examples => https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
#             https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/


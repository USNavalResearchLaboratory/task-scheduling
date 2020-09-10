
import gym
import numpy as np
import pickle
import kneed
import os
import sys, os
import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../anotherproject")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from RL_Priority.env_priority import PriorityQueue, PriorityQueueDiscrete
from RL_Priority.env_priority import DiscretizedObservationWrapper
from collections import defaultdict
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env


def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


env = PriorityQueueDiscrete()
check_env(env)


model = A2C(MlpPolicy, env, verbose=1)
mean_reward_before_train = evaluate(model, num_steps=10000)

model.learn(total_timesteps=25000)

mean_reward = evaluate(model, num_steps=10000)
a = 1



# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
#
# env = gym.make('CartPole-v1')
# # Optional: PPO2 requires a vectorized environment to run
# # the env is now wrapped automatically when passing it to the constructor
# # env = DummyVecEnv([lambda: env])
#
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
#
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param


# env = PriorityQueue()
# # Instantiate the agent
# model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# # Train the agent
# model.learn(total_timesteps=int(2e5))
#
#
# env = gym.make("CartPole-v0")
#
# env = DiscretizedObservationWrapper(
#     env,
#     n_bins=8,
#     low=[-2.4, -2.0, -0.42, -3.5],
#     high=[2.4, 2.0, 0.42, 3.5]
# )
#
#
# actions = range(env.action_space)
#
# def update_Q(s, r, a, s_next, done):
#     max_q_next = max([Q[s_next, a] for a in actions])
#     # Do not include the next state's value if currently at the terminal state.
#     Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])


def main():

    policy = 'priority' # 'first', 'random'

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

            if policy == 'priority':
                # Reassess Track Priorities
                job = Queue.job
                for jj in range(len(job)):
                    job[jj].Priority = job[jj](timeSec)
                priority = np.array([task.Priority for task in job])
                priority_Idx = np.argsort(-1*priority, kind='mergesort')
                N = Queue.env_config.get('N')
                action = np.zeros(N,dtype='int64')
                possible_actions = np.arange(0,Queue.M)
                for jj in range(N):
                    index = np.where(possible_actions == priority_Idx[jj])
                    action[jj] = index[0]
                    possible_actions = np.delete(possible_actions, index[0])
                    # priority_Idx = np.delete(priority_Idx, 0)
                    # priority_Idx = priority_Idx - 1 # Decrement by 1 to capture new indicies
            elif policy == 'random':
                action = Queue.action_space.sample()  # Insert your policy here
            elif policy == 'first':
                N = Queue.env_config.get("N")
                action = np.zeros(N, dtype='int64')  # Always choose first action available
            else:
                print('Policy unrecognized')

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
    mean_reward = np.mean(reward, axis=0)
    plt.plot(RP*np.arange(len(mean_reward)),  np.mean(reward, axis=0))
    plt.xlabel('Time (s)')
    plt.show()

if __name__ == '__main__':
    main()

if 0:
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






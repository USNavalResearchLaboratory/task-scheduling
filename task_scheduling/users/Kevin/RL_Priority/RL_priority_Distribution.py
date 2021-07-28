import numpy as np
import pickle
import sys, os
import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../anotherproject")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from users.Kevin.RL_Priority.env_priority import PriorityQueue


# best_mean_reward, n_steps = -np.inf, 0
#
# def callback(_locals, _globals):
#   """
#   Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
#   :param _locals: (dict)
#   :param _globals: (dict)
#   """
#   global n_steps, best_mean_reward
#   # Print stats every 1000 calls
#   if (n_steps + 1) % 100 == 0:
#       # Evaluate policy performance
#       x, y = ts2xy(load_results(log_dir), 'timesteps')
#       if len(x) > 0:
#           mean_reward = np.mean(y[-100:])
#           print(x[-1], 'timesteps')
#           print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
#
#           # New best model, you could save the agent here
#           if mean_reward > best_mean_reward:
#               best_mean_reward = mean_reward
#               # Example for saving best model
#               print("Saving new best model")
#               _locals['self'].save(log_dir + 'best_model.pkl')
#   n_steps += 1
#   return False
#
#
# def moving_average(values, window):
#     """
#     Smooth values by doing a moving average
#     :param values: (numpy array)
#     :param window: (int)
#     :return: (numpy array)
#     """
#     weights = np.repeat(1.0, window) / window
#     return np.convolve(values, weights, 'valid')
#
# def plot_results(log_folder, title='Learning Curve', figNum = 1):
#     """
#     plot the results
#
#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     x, y = ts2xy(load_results(log_folder), 'timesteps')
#     y = moving_average(y, window=1000)
#     # Truncate x
#     x = x[len(x) - len(y):]
#
#     fig = plt.figure(figNum)
#     # fig = plt.figure(title=title)
#     plt.plot(x, y)
#     plt.xlabel('Number of Timesteps')
#     plt.ylabel('Rewards')
#     plt.title(title + " Smoothed")
#     plt.show()
#
#
# def evaluate(model, num_steps=1000):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_steps: (int) number of timesteps to evaluate it
#     :return: (float) Mean reward for the last 100 episodes
#     """
#     episode_rewards = [0.0]
#     obs = env.reset()
#     for i in range(num_steps):
#         # _states are only useful when using LSTM policies
#         action, _states = model.predict(obs)
#
#         obs, reward, done, info = env.step(action)
#
#         # Stats
#         episode_rewards[-1] += reward
#         if done:
#             obs = env.reset()
#             episode_rewards.append(0.0)
#     # Compute mean reward for the last 100 episodes
#     mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
#     print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
#
#     return mean_100ep_reward
#
#
# # env = PriorityQueueDiscrete()
#
#
# # Create log dir
# log_dir_A2C = "./tmp/A2C/"
# os.makedirs(log_dir_A2C, exist_ok=True)
#
# log_dir_PPO2 = "./tmp/PPO2/"
# os.makedirs(log_dir_PPO2, exist_ok=True)
#
# # env = gym.make('LunarLanderContinuous-v2')
#
# env_A2C = TaskAssignmentDiscrete()
# env_A2C = Monitor(env_A2C, log_dir_A2C, allow_early_resets=True)
#
# env_PPO2 = TaskAssignmentDiscrete()
# env_PPO2 = Monitor(env_PPO2, log_dir_PPO2, allow_early_resets=True)
#
#
# # env.reset()
# # action = env.action_space.sample()
# # env.step(action)
#
#
# # check_env(env)
#
#
# model = A2C(MlpPolicy, env_A2C, verbose=1, learning_rate=0.005)
# model_PPO2 = PPO2(MlpPolicy, env_PPO2, verbose=1, learning_rate=0.005)
# # model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./log_tensorboard/")
#
#
#
#
# model.learn(total_timesteps=100000)
# model.save(log_dir_A2C + 'A2C')
# plot_results(log_dir_A2C, figNum=1)
#
# model_PPO2.learn(total_timesteps=100000)
# model_PPO2.save(log_dir_PPO2 + 'PPO2')
# plot_results(log_dir_PPO2, figNum=2)
#
# # model.learn(total_timesteps=200, tb_log_name="first_run")
# # model.learn(total_timesteps=200, callback=callback)
#
# # std_reward = np.std(episode_rewards)
# # mean_reward_before_train = evaluate(model, num_steps=1000)
#
#
# episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True)
# episode_rewards_train, episode_lengths_train = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True)
#
#
# mean_reward_rolling = moving_average(episode_rewards, 10)
# mean_reward_rolling_train = moving_average(episode_rewards_train, 10)
#
# plt.figure(2)
# plt.clf()
# plt.plot(mean_reward_rolling, label='Before Train')
# plt.plot(mean_reward_rolling_train, label='After Train')
# plt.legend
# plt.show()
# # mean_reward = evaluate(model, num_steps=10000)
#
#
#
# a = 1


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

##

# Q = defaultdict(float)
# gamma = 0.99  # Discounting factor
# alpha = 0.5  # soft update param
#
#
# # env = PriorityQueue()
# # # Instantiate the agent
# # model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# # # Train the agent
# # model.learn(total_timesteps=int(2e5))
# #
# #
# # env = gym.make("CartPole-v0")
# #
# # env = DiscretizedObservationWrapper(
# #     env,
# #     n_bins=8,
# #     low=[-2.4, -2.0, -0.42, -3.5],
# #     high=[2.4, 2.0, 0.42, 3.5]
# # )
# #
# #
# # actions = range(env.action_space)
# #
# # def update_Q(s, r, a, s_next, done):
# #     max_q_next = max([Q[s_next, a] for a in actions])
# #     # Do not include the next state's value if currently at the terminal state.
# #     Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])
#
#
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

if 1:
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






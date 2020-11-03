import time
import os
from copy import deepcopy
from types import MethodType
from math import factorial
import dill

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Space, Discrete

from util.plot import plot_task_losses
from util.generic import seq2num, num2seq
from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift
from task_scheduling.environments import SeqTaskingEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as MlpPolicyQ
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from stable_baselines.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


def evaluate(model, env, num_steps=1000):
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


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

    weights = np.ones(50) / 50
    y = np.convolve(y, weights, 'valid')

    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def train_agent_sb(problem_gen, env_cls=SeqTaskingEnv, env_params=None,
                   save=False, save_dir=None, algorithm='DQN', policy='mlp', timesteps=1000):
    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(problem_gen, **env_params)

    # Create log dir
    log_dir = "../tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    if (algorithm == 'DQN') and (policy == 'mlp'): 
        agent = DQN(MlpPolicyQ, env)
    elif (algorithm == 'PPO2') and (policy == 'mlp'):
        agent = PPO2(MlpPolicy, env)
    elif (algorithm == 'A2C') and (policy == 'mlp'):
        agent = A2C(MlpPolicy, env)
        env = DummyVecEnv([lambda: env])

    else:
        raise NotImplementedError

    print('training stable baselines agent . . . good luck')
    mean_reward_before_train = evaluate(agent, env, num_steps=10000)

    if save:
        # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../agents/',
        #                                  name_prefix=algorithm + policy)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        agent.learn(total_timesteps=timesteps, callback=callback)
        # agent.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=100, tb_log_name="DQN")

    else:
        agent.learn(total_timesteps=timesteps)

    plot_results(log_dir)
    mean_reward = evaluate(agent, env, num_steps=10000)

    return wrap_agent(env, agent)


def load_agent_sb(load_dir, env):
    """Loads agent and environment, returns wrapped scheduling function."""
    if ('DQN' in load_dir) and ('mlp' in load_dir):
        agent = DQN(MlpPolicyQ, env, verbose=1)
        agent.load(load_dir, env)
    else:
        raise NotImplementedError

    return wrap_agent(env, agent)


def main():

    problem_gen = RandomProblem.relu_drop(n_tasks=6, n_ch=2)

    features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                         ('release time', lambda task: task.t_release,
                          (0., problem_gen.task_gen.param_lims['t_release'][1])),
                         ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                         ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                         ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                         ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
    # features = None

    # def seq_encoding(self, n):
    #     return [0] if n in self.node.seq else [1]

    def seq_encoding(self, n):
        out = np.zeros(self.n_tasks)
        if n in self.node.seq:
            out[self.node.seq.index(n)] = 1
        return out

    # seq_encoding = 'binary'
    # seq_encoding = None

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.tasks[n].t_release
            # return 1 if self.tasks[n].l_drop == 0. else 0
            # return self.tasks[n].l_drop / self.tasks[n].t_drop

    # sort_func = 't_release'

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': False,
                  'action_type': 'int',
                  # 'seq_encoding': seq_encoding,
                  }
    env = SeqTaskingEnv(problem_gen, **env_params)
    # env = StepTaskingEnv(problem_gen, **env_params)

#    agent = DQN(MlpPolicyQ, env, verbose=1)
#    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../agents/',
#                                         name_prefix='rl_model')
#    agent.learn(total_timesteps=1000,callback=checkpoint_callback)
#    del agent 
#    agent = DQN(MlpPolicyQ, env, verbose=1)
#    agent.load('../agents/rl_model_1000_steps.zip', env)


if __name__ == '__main__':
    main()

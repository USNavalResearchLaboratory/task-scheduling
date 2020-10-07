import os
import time
import dill

import numpy as np

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import plot_results
from stable_baselines import DQN, PPO2, A2C

from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift
from environments import BaseTaskingEnv, SeqTaskingEnv, StepTaskingEnv

np.set_printoptions(precision=2)


# Agents
class RandomAgent:      # TODO: subclass or keep duck-typing?
    """Uniformly random action selector."""
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        action_space = self.env.action_space
        # action_space = self.env.infer_action_space(obs)
        return action_space.sample(), None       # randomly selected action

    def learn(self, *args, **kwargs):
        pass

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env


class LearningScheduler:
    model_cls_dict = {'DQN': DQN, 'PPO2': PPO2, 'A2C': A2C}
    log_dir = '../logs/SB_train'

    def __init__(self, model, env=None):
        self.model = model

        self.do_monitor = True  # TODO: make init parameter?
        if env is not None:
            self.env = env

    @property
    def env(self):
        return self.model.get_env()

    @env.setter
    def env(self, env):
        if isinstance(env, BaseTaskingEnv):
            if self.do_monitor:
                env = Monitor(env, self.log_dir)
            self.model.set_env(env)
        else:
            raise TypeError("Environment must be an instance of BaseTaskingEnv.")

    def __call__(self, tasks, ch_avail):
        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)
        done = False
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def learn(self, n_episodes=0):
        self.model.learn(total_timesteps=n_episodes * self.env.steps_per_episode)
        if self.do_monitor:
            plot_results([self.log_dir], num_timesteps=None, xaxis='timesteps', task_name='Training history')

    def save(self, save_path=None):
        if save_path is None:
            cls_str = self.model.__class__.__name__
            save_path = f"temp/{cls_str}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        save_path = '../agents/' + save_path
        # os.makedirs(save_path, exist_ok=True)

        self.model.save(save_path)

        # with open(save_path + '/env', 'wb') as file:    # TODO: save Env?
        #     dill.dump(self.env, file)    # save environment

    @classmethod
    def load(cls, load_path, env=None, model_cls=None):
        if model_cls is None:
            cls_str = load_path.split('/')[-1].split('_')[0]
            model_cls = cls.model_cls_dict[cls_str]

        model = model_cls.load('../agents/' + load_path)
        return cls(model, env)

    @classmethod
    def load_from_gen(cls, load_path, problem_gen, env_cls=StepTaskingEnv, env_params=None, model_cls=None):
        env = cls.gen_to_env(problem_gen, env_cls, env_params)
        return cls.load(load_path, env, model_cls)

    @classmethod
    def from_gen(cls, model, problem_gen, env_cls=StepTaskingEnv, env_params=None):
        env = cls.gen_to_env(problem_gen, env_cls, env_params)
        return cls(model, env)

    @classmethod
    def train_agent(cls, model, problem_gen, env_cls=SeqTaskingEnv, env_params=None, n_episodes=0,
                    save=False, save_path=None):

        env = cls.gen_to_env(problem_gen, env_cls, env_params)

        if model is None or model == 'random':      # TODO: improve
            model = RandomAgent(env)
        elif model == 'DQN':
            model = DQN('MlpPolicy', env, verbose=1)

        scheduler = cls(model, env)
        scheduler.learn(n_episodes)
        if save:
            scheduler.save(save_path)

        return scheduler

    @staticmethod
    def gen_to_env(problem_gen, env_cls, env_params):
        if env_params is None:
            env_params = {}
        return env_cls(problem_gen, **env_params)


# def train_agent(problem_gen, n_episodes=0, env_cls=SeqTaskingEnv, env_params=None,
#                 model=None, save=False, save_path=None):
#     """
#     Train a reinforcement learning agent.
#
#     Parameters
#     ----------
#     problem_gen : generators.scheduling_problems.Base
#         Scheduling problem generation object.
#     n_episodes : int
#         Number of complete environment episodes used for training.
#     env_cls : class, optional
#         Gym environment class.
#     env_params : dict, optional
#         Parameters for environment initialization.
#     model : BaseRLModel or str, optional
#         Reinforcement learning agent.
#     save : bool
#         If True, the agent and environment are serialized.
#     save_path : str, optional
#         String representation of sub-directory to save to.
#
#     Returns
#     -------
#     function
#         Wrapped agent. Takes tasks and channel availabilities and produces task execution times/channels.
#
#     """
#
#     # Create environment
#     if env_params is None:
#         env_params = {}
#
#     env = env_cls(problem_gen, **env_params)
#
#     # Train agent
#     if model is None or model == 'random':
#         model = RandomAgent(env)
#     elif model == 'DQN':
#         model = DQN('MlpPolicy', env, verbose=1)
#
#     model.learn(total_timesteps=n_episodes * env.steps_per_episode)
#
#     # Save agent and environment
#     if save:
#         if save_path is None:
#             cls_str = model.__class__.__name__
#             save_path = f"../agents/temp/{cls_str}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
#         else:
#             save_path = '../agents/' + save_path
#
#         model.save(save_path)
#
#     return model
#     # return env.wrap_model(model)


# def load_agent(load_path, model_cls=None):
#     """Loads agent and environment, returns wrapped scheduling function."""
#
#     if model_cls is None:
#         cls_str = load_path.split('/')[-1].split('_')[0]
#         cls_dict = {'DQN': DQN, 'PPO2': PPO2, 'A2C': A2C}
#         model_cls = cls_dict[cls_str]
#
#     model = model_cls.load('../agents/' + load_path)
#     return model
#     # return env.wrap_model(model)


# def wrap_agent(env, model):
#     """Generate scheduling function by running an agent on a single environment episode."""
#
#     def scheduling_agent(tasks, ch_avail):
#         obs = env.reset(tasks, ch_avail)
#         done = False
#         while not done:
#             action, _states = model.predict(obs)
#             obs, reward, done, info = env.step(action)
#
#         return env.node.t_ex, env.node.ch_ex
#
#     return scheduling_agent


# def wrap_agent_run_lim(env, agent):     # FIXME
#     """Generate scheduling function by running an agent on a single environment episode, enforcing max runtime."""
#
#     def scheduling_agent(tasks, ch_avail, max_runtime):
#
#         t_run = time.perf_counter()
#
#         obs = env.reset(tasks, ch_avail)
#         done = False
#         while not done:
#             action, _states = agent.predict(obs)
#             obs, reward, done, info = env.step(action)
#
#         runtime = time.perf_counter() - t_run
#         if runtime >= max_runtime:
#             raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")
#
#         return env.node.t_ex, env.node.ch_ex
#
#     return scheduling_agent


def main():
    problem_gen = RandomProblem.relu_drop(n_tasks=4, n_ch=2)

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

    env_cls = SeqTaskingEnv
    # env_cls = StepTaskingEnv

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': False,
                  'action_type': 'int',
                  # 'seq_encoding': seq_encoding,
                  }

    env = env_cls(problem_gen, **env_params)

    s = LearningScheduler.train_agent('DQN', problem_gen, env_cls, env_params, n_episodes=10000,
                                      save=False, save_path=None)


    # model = RandomAgent(env)
    # model = DQN('MlpPolicy', env, verbose=1)
    # model.learn(10)

    # scheduler = train_agent(problem_gen, n_episodes=10, env_cls=env_cls, env_params=env_params, agent=agent)

    # obs = env.reset()
    # done = False
    # while not done:
    #     print(obs)
    #     # print(env.sorted_index)
    #     # print(env.node.seq)
    #     # print(env.tasks)
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, reward, done, info = env.step(action)
    #     print(reward)


if __name__ == '__main__':
    main()

import time
from collections import namedtuple
from pathlib import Path
import dill

import numpy as np

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import plot_results
from stable_baselines import DQN, PPO2, A2C
# from stable_baselines.common.vec_env import DummyVecEnv

from task_scheduling.learning import environments as envs

np.set_printoptions(precision=2)

pkg_path = Path.cwd()
log_path = pkg_path / 'logs'
agent_path = pkg_path / 'agents'


# class DummyVecTaskingEnv(DummyVecEnv):
#     def reset(self, *args, **kwargs):
#         for env_idx in range(self.num_envs):
#             obs = self.envs[env_idx].reset(*args, **kwargs)
#             self._save_obs(env_idx, obs)
#         return self._obs_from_buf()


# Agents
class RandomAgent:
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


# Schedulers
class ReinforcementLearningScheduler:
    log_dir = log_path / 'SB_train'

    _default_tuple = namedtuple('ModelDefault', ['cls', 'kwargs'])
    model_defaults = {'Random': _default_tuple(RandomAgent, {}),
                      'DQN': _default_tuple(DQN, {'policy': 'MlpPolicy', 'verbose': 1}),
                      'DQN_LN': _default_tuple(DQN, {'policy': 'LnMlpPolicy', 'verbose': 1}),
                      'CNN': _default_tuple(DQN, {'policy': 'CnnPolicy', 'verbose': 1}),
                      'PPO2': _default_tuple(PPO2, {}),
                      'A2C': _default_tuple(A2C, {'policy': 'MlpPolicy', 'verbose': 1}),
                      }

    def __init__(self, model, env=None):
        self.model = model

        self.do_monitor = True  # TODO: make init parameter?
        if env is not None:
            self.env = env  # invokes setter

    # Environment access
    @property
    def env(self):
        return self.model.get_env()

    @env.setter
    def env(self, env):
        if isinstance(env, envs.BaseTasking):
            if self.do_monitor:
                env = Monitor(env, str(self.log_dir))
            self.model.set_env(env)
        else:
            raise TypeError("Environment must be an instance of BaseTasking.")

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Iterable of task_scheduling.tasks.Base
        ch_avail : Iterable of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        ensure_valid = isinstance(self.env, envs.StepTasking) and not self.env.do_valid_actions

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)    # kwargs required for wrapped Monitor environment
        done = False
        while not done:
            # action, _states = self.model.predict(obs)

            prob = self.model.action_probability(obs)
            if ensure_valid:
                prob = self.env.mask_probability(prob)
            action = prob.argmax()

            obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def learn(self, n_episodes=0):
        """
        Train learning model.

        Parameters
        ----------
        n_episodes : int
            Number of complete tasking episode to train on.

        """

        self.model.learn(total_timesteps=n_episodes * self.env.steps_per_episode)

        if self.do_monitor:
            plot_results([self.log_dir], num_timesteps=None, xaxis='timesteps', task_name='Training history')

    def save(self, save_path=None):
        if save_path is None:
            cls_str = self.model.__class__.__name__
            save_path = f"temp/{cls_str}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        save_path = agent_path / save_path
        save_path.parent.mkdir(exist_ok=True)

        self.model.save(str(save_path))

        with save_path.parent.joinpath(save_path.stem + '_env').open(mode='wb') as fid:
            env_ = self.env.env     # extract base env from Monitor
            dill.dump(env_, fid)  # save environment

    @classmethod
    def load(cls, load_path, env=None, model_cls=None):
        if model_cls is None:
            cls_str = load_path.split('/')[-1].split('_')[0]
            model_cls = cls.model_defaults[cls_str].cls
        elif isinstance(model_cls, str):
            model_cls = cls.model_defaults[model_cls].cls

        load_path = agent_path / load_path
        model = model_cls.load(str(load_path))

        try:
            with load_path.parent.joinpath(load_path.stem + '_env').open(mode='rb') as fid:
                env = dill.load(fid)
        except FileNotFoundError:
            pass

        return cls(model, env)

    # @classmethod
    # def load_from_gen(cls, load_path, problem_gen, env_cls=StepTasking, env_params=None, model_cls=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls.load(load_path, env, model_cls)

    # @classmethod
    # def from_gen(cls, model, problem_gen, env_cls=StepTasking, env_params=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls(model, env)

    @classmethod
    def train_from_gen(cls, problem_gen, env_cls=envs.SeqTasking, env_params=None, model_cls=None, model_params=None,
                       n_episodes=0, save=False, save_path=None):
        """
        Create and train a reinforcement learning scheduler.

        Parameters
        ----------
        problem_gen : generators.scheduling_problems.Base
            Scheduling problem generation object.
        env_cls : class, optional
            Gym environment class.
        env_params : dict, optional
            Parameters for environment initialization.
        model_cls : class, optional
            RL model class.
        model_params : dict, optional
            RL model parameters.
        n_episodes : int
            Number of complete environment episodes used for training.
        save : bool, optional
            If True, the agent and environment are serialized.
        save_path : str, optional
            String representation of sub-directory to save to.

        Returns
        -------
        ReinforcementLearningScheduler

        """

        env = env_cls.from_problem_gen(problem_gen, env_params)

        if model_params is None:
            model_params = {}

        if model_cls is None:
            model_cls, model_params = cls.model_defaults['Random']
        elif isinstance(model_cls, str):
            model_cls, model_params_ = cls.model_defaults[model_cls]
            model_params_.update(model_params)
            model_params = model_params_

        model = model_cls(env=env, **model_params)

        scheduler = cls(model, env)
        scheduler.learn(n_episodes)
        if save:
            scheduler.save(save_path)

        return scheduler

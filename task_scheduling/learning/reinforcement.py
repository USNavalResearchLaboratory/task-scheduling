from collections import namedtuple
from pathlib import Path
# import dill

from stable_baselines3 import DQN, A2C
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

# from task_scheduling.learning import environments as envs
from task_scheduling.learning.base import Base as BaseLearningScheduler
from task_scheduling.learning.supervised.torch import reset_weights


# from stable_baselines.common.vec_env import DummyVecEnv


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

    def predict(self, _obs):
        action_space = self.env.action_space
        # action_space = self.env.infer_action_space(obs)
        return action_space.sample(), None  # randomly selected action

    def learn(self, *args, **kwargs):
        pass

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env


# Schedulers
class StableBaselinesScheduler(BaseLearningScheduler):
    log_dir = Path.cwd() / 'logs/learn' / 'sb'

    _default_tuple = namedtuple('ModelDefault', ['cls', 'kwargs'])
    model_defaults = {'Random': _default_tuple(RandomAgent, {}),
                      'DQN_MLP': _default_tuple(DQN, {'policy': 'MlpPolicy', 'verbose': 1}),
                      'DQN_LN': _default_tuple(DQN, {'policy': 'LnMlpPolicy', 'verbose': 1}),
                      'DQN_CNN': _default_tuple(DQN, {'policy': 'CnnPolicy', 'verbose': 1}),
                      # 'PPO2': _default_tuple(PPO2, {}),
                      'A2C': _default_tuple(A2C, {'policy': 'MlpPolicy', 'verbose': 1}),
                      }

    do_monitor = False

    def __init__(self, env, model, learn_params=None):  # TODO: remove `env` arg? Change inheritance?
        super().__init__(env, model, learn_params)

        # self.model = model
        # self.env = env  # invokes setter

        # self.steps_per_episode = env.steps_per_episode  # FIXME: weak hack...

    @classmethod
    def make_model(cls, env, model_cls, model_params, learn_params=None):
        model = model_cls(env=env, **model_params)
        return cls(env, model, learn_params)

    # @property
    # def env(self):
    #     return self.model.get_env()
    #
    # @env.setter
    # def env(self, env):
    #     if isinstance(env, envs.BaseTasking):
    #         if self.do_monitor:
    #             env = Monitor(env, str(self.log_dir))
    #         self.model.set_env(env)
    #     elif env is not None:
    #         raise TypeError("Environment must be an instance of BaseTasking.")

    # def predict_prob(self, obs):
    #     return self.model.action_probability(obs)  # TODO: need `env.env_method` to access my reset?

    def predict(self, obs):
        action, _state = self.model.predict(obs)  # TODO: include state output, change base class?
        return action

    def learn(self, n_gen_learn, verbose=0):
        # TODO: consider using `eval_env` argument to pass original env for `model.learn` call

        # steps_per_episode = 8
        # steps_per_episode = self.steps_per_episode
        steps_per_episode = self.env.steps_per_episode  # TODO: breaks due to env vectorization
        self.model.learn(total_timesteps=n_gen_learn * steps_per_episode)

        if self.do_monitor:
            plot_results([str(self.log_dir)], num_timesteps=None, x_axis='timesteps', task_name='Training history')

    def reset(self):
        self.model.policy.apply(reset_weights)

    # def save(self, save_path=None):
    #     if save_path is None:
    #         cls_str = self.model.__class__.__name__
    #         save_path = f"temp/{cls_str}_{NOW_STR}"
    #
    #     save_path = Path.cwd() / 'agents' / save_path
    #     save_path.parent.mkdir(exist_ok=True)
    #
    #     self.model.save(str(save_path))
    #
    #     with save_path.parent.joinpath(save_path.stem + '_env').open(mode='wb') as fid:
    #         env_ = self.env.env  # extract base env from Monitor
    #         dill.dump(env_, fid)  # save environment
    #
    # @classmethod
    # def load(cls, load_path, env=None, model_cls=None):
    #     if model_cls is None:
    #         cls_str = load_path.split('/')[-1].split('_')[0]
    #         model_cls = cls.model_defaults[cls_str].cls
    #     elif isinstance(model_cls, str):
    #         model_cls = cls.model_defaults[model_cls].cls
    #
    #     load_path = Path.cwd() / 'agents' / load_path
    #     model = model_cls.load(str(load_path))
    #
    #     try:
    #         with load_path.parent.joinpath(load_path.stem + '_env').open(mode='rb') as fid:
    #             env = dill.load(fid)
    #     except FileNotFoundError:
    #         pass
    #
    #     return cls(model, env)

    # @classmethod
    # def load_from_gen(cls, load_path, problem_gen, env_cls=StepTasking, env_params=None, model_cls=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls.load(load_path, env, model_cls)

    # @classmethod
    # def from_gen(cls, model, problem_gen, env_cls=StepTasking, env_params=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls(model, env)

    # @classmethod
    # def train_from_gen(cls, problem_gen, env_cls=envs.SeqTasking, env_params=None, model_cls=None, model_params=None,
    #                    n_episodes=0, save=False, save_path=None):
    #     """
    #     Create and train a reinforcement learning scheduler.
    #
    #     Parameters
    #     ----------
    #     problem_gen : generators.scheduling_problems.Base
    #         Scheduling problem generation object.
    #     env_cls : class, optional
    #         Gym environment class.
    #     env_params : dict, optional
    #         Parameters for environment initialization.
    #     model_cls : class, optional
    #         RL model class.
    #     model_params : dict, optional
    #         RL model parameters.
    #     n_episodes : int
    #         Number of complete environment episodes used for training.
    #     save : bool, optional
    #         If True, the agent and environment are serialized.
    #     save_path : str, optional
    #         String representation of sub-directory to save to.
    #
    #     Returns
    #     -------
    #     StableBaselinesScheduler
    #
    #     """
    #
    #     # Create environment
    #     if env_params is None:
    #         env = env_cls(problem_gen)
    #     else:
    #         env = env_cls(problem_gen, **env_params)
    #
    #     # Create model
    #     if model_params is None:
    #         model_params = {}
    #
    #     if model_cls is None:
    #         model_cls, model_params = cls.model_defaults['Random']
    #     elif isinstance(model_cls, str):
    #         model_cls, model_params_ = cls.model_defaults[model_cls]
    #         model_params_.update(model_params)
    #         model_params = model_params_
    #
    #     model = model_cls(env=env, **model_params)
    #
    #     # Create and train scheduler
    #     scheduler = cls(model, env)
    #     scheduler.learn(n_episodes)
    #     if save:
    #         scheduler.save(save_path)
    #
    #     return scheduler

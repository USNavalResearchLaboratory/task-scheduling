from collections import namedtuple

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import DQN, A2C, PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from torch import nn


# from task_scheduling.mdp import environments as envs
from task_scheduling.mdp.base import BaseLearning as BaseLearningScheduler
from task_scheduling.mdp.supervised.torch import reset_weights, valid_logits


# class DummyVecTaskingEnv(DummyVecEnv):
#     def reset(self, *args, **kwargs):
#         for env_idx in range(self.num_envs):
#             obs = self.envs[env_idx].reset(*args, **kwargs)
#             self._save_obs(env_idx, obs)
#         return self._obs_from_buf()

# TODO: use agents that can exploit expert knowledge

_default_tuple = namedtuple('ModelDefault', ['cls', 'params'], defaults={})


class StableBaselinesScheduler(BaseLearningScheduler):
    _learn_params_default = {
        'max_epochs': 1,
    }

    model_defaults = {
        # 'Random': _default_tuple(RandomAgent, {}),
        'DQN_MLP': _default_tuple(DQN, {'policy': 'MlpPolicy', 'verbose': 1}),
        'DQN_LN': _default_tuple(DQN, {'policy': 'LnMlpPolicy', 'verbose': 1}),
        'DQN_CNN': _default_tuple(DQN, {'policy': 'CnnPolicy', 'verbose': 1}),
        'A2C': _default_tuple(A2C, {'policy': 'MlpPolicy', 'verbose': 1}),
        'PPO': _default_tuple(PPO, {'policy': 'MlpPolicy', 'verbose': 1})
    }

    do_monitor = False

    def __init__(self, env, model, learn_params=None):  # TODO: remove `env` arg? Change inheritance?
        super().__init__(env, model, learn_params)

        # self.model = model
        # self.env = env  # invokes setter

    @classmethod
    def make_model(cls, env, model_cls, model_params=None, learn_params=None):
        if model_params is None:
            model_params = {}
        if isinstance(model_cls, str):
            model_cls, _params = cls.model_defaults[model_cls]
            model_params = _params | model_params

        model = model_cls(env=env, **model_params)
        return cls(env, model, learn_params)

    # @property
    # def env(self):
    #     return self.model.get_env()
    #
    # @env.setter
    # def env(self, env):
    #     if isinstance(env, envs.BaseEnv):
    #         if self.do_monitor:
    #             env = Monitor(env, str(self.log_dir))
    #         self.model.set_env(env)
    #     elif env is not None:
    #         raise TypeError("Environment must be an instance of BaseEnv.")

    # def predict_prob(self, obs):
    #     return self.model.action_probability(obs)  # TODO: need `env.env_method` to access my reset?

    def predict(self, obs):
        action, _state = self.model.predict(obs, deterministic=True)
        return action

    def learn(self, n_gen_learn, verbose=0):
        # TODO: consider using `eval_env` argument to pass original env for `model.learn` call

        total_timesteps = self.learn_params['max_epochs'] * n_gen_learn * self.env.steps_per_episode
        self.model.learn(total_timesteps)

        # if self.do_monitor:
        #     plot_results([str(self.log_dir)], num_timesteps=None, x_axis='timesteps', task_name='Training history')

    def reset(self):
        self.model.policy.apply(reset_weights)

    # def save(self, save_path=None):
    #     if save_path is None:
    #         cls_str = self.model.__class__.__name__
    #         save_path = f"temp/{cls_str}_{NOW_STR}"
    #
    #     save_path = Path.cwd() / 'agents' / save_path
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
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
    # def load_from_gen(cls, load_path, problem_gen, env_cls=Index, env_params=None, model_cls=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls.load(load_path, env, model_cls)

    # @classmethod
    # def from_gen(cls, model, problem_gen, env_cls=Index, env_params=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls(model, env)

    # @classmethod
    # def train_from_gen(cls, problem_gen, env_cls=envs.Seq, env_params=None, model_cls=None, model_params=None,
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


# FIXME
# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the feature extractor.
#
#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """
#
#     def __init__(
#         self,
#         feature_dim: int,
#         last_layer_dim_pi: int = 64,
#         last_layer_dim_vf: int = 64,
#     ):
#         super(CustomNetwork, self).__init__()
#
#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf
#
#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
#         )
#
#     def forward(self, features):
#         """
#         :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.policy_net(features), self.value_net(features)


# class CustomFlattenExtractor(FlattenExtractor):
#     def __init__(self, observation_space: Box):
#         # create new space for only the task feature observation
#         low, high = observation_space.low[..., 1:], observation_space.high[..., 1:]
#         observation_space = Box(low, high, dtype=observation_space.dtype)
#         super().__init__(observation_space)


class FlattenDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)[:, ::2]


class CustomCombinedExtractor(BaseFeaturesExtractor):  # TODO: generalize for my SL modules!!!
    # space_keys = ['ch_avail', 'seq', 'tasks']
    space_keys = ['seq', 'tasks']  # TODO: ignores chan info for simplicity

    def __init__(self, observation_space: spaces.Dict):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key in self.space_keys:
            space = observation_space[key]
            if key == 'seq':  # override one-hot encoding
                extractors[key] = FlattenDownsample()
                total_concat_size += spaces.utils.flatdim(space)
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(space)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class ValidActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, infer_valid_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        if callable(infer_valid_mask):
            self.infer_valid_mask = infer_valid_mask
        else:
            self.infer_valid_mask = lambda obs: np.ones(self.observation_space.shape)

    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = CustomNetwork(self.features_dim)

    def _get_action_dist_from_latent_valid(self, obs, latent_pi, _latent_sde=None):
        mean_actions = self.action_net(latent_pi)
        mean_actions = valid_logits(mean_actions, self.infer_valid_mask(obs))  # mask out invalid actions
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_valid(obs, latent_pi, latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        latent_pi, _, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent_valid(obs, latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent_valid(obs, latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

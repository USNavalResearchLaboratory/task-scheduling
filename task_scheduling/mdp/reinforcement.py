"""Reinforcement learning schedulers and custom policies."""

import math
import time
from collections import namedtuple
from functools import partial
from pathlib import Path

import dill
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import configure_logger, obs_as_tensor, safe_mean
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn import functional
from tqdm import tqdm, trange

from task_scheduling.base import get_now
from task_scheduling.mdp.base import BaseLearning as BaseLearningScheduler
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.util import (
    build_cnn,
    build_mlp,
    flatten_rollouts,
    make_dataloaders,
    make_dataloaders_dict,
    reset_weights,
    reward_to_go,
    valid_logits,
)

_default_tuple = namedtuple("ModelDefault", ["cls", "params"], defaults={})


# TODO: allow general vectorized environments


class StableBaselinesScheduler(BaseLearningScheduler):
    """
    Base class for learning schedulers.

    Parameters
    ----------
    env : BaseEnv
        OpenAi gym environment.
    model
        The learning object.
    learn_params : dict, optional
        Parameters used by the `learn` method.

    """

    _learn_params_default = {"frac_val": 0.0, "max_epochs": 1, "eval_callback_kwargs": None}

    model_defaults = {
        "DQN": _default_tuple(DQN, {"policy": "MlpPolicy", "verbose": 1}),
        "A2C": _default_tuple(A2C, {"policy": "MlpPolicy", "verbose": 1}),
        "PPO": _default_tuple(PPO, {"policy": "MlpPolicy", "verbose": 1}),
    }

    def __init__(self, env, model, learn_params=None):
        super().__init__(env, model, learn_params)

        if self.learn_params["eval_callback_kwargs"] is None:
            self.learn_params["eval_callback_kwargs"] = {}

    @classmethod
    def make_model(cls, env, model_cls, model_kwargs=None, learn_params=None):
        """Construct scheduler from Stable-Baselines3 model specification."""
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(model_cls, str):
            model_cls, _kwargs = cls.model_defaults[model_cls]
            model_kwargs = _kwargs | model_kwargs

        model = model_cls(env=env, **model_kwargs)
        return cls(env, model, learn_params)

    @property
    def env(self):
        # return self.model.get_env()
        return self.model.get_env().envs[0].env  # unwrap vectorized, monitored environment

    @env.setter
    def env(self, env):
        # self.model.set_env(env)
        self.model.get_env().envs[0].env = env

    def predict(self, obs):
        """
        Take an action given an observation.

        Parameters
        ----------
        obs : array_like
            Observation.

        Returns
        -------
        int or array_like
            Action.

        """
        action, _state = self.model.predict(obs, deterministic=True)
        return action

    def learn(self, n_gen, verbose=0):
        """
        Learn from the environment.

        Parameters
        ----------
        n_gen : int
            Number of problems to generate data from.
        verbose : int, optional
            Progress print-out level.

        """
        n_gen_val = math.floor(n_gen * self.learn_params["frac_val"])
        n_gen_train = n_gen - n_gen_val

        total_timesteps = self.learn_params["max_epochs"] * n_gen_train * self.env.action_space.n

        if n_gen_val > 0:
            problem_gen_val = self.env.problem_gen.split(n_gen_val, shuffle=True, repeat=True)
            eval_env = Index(
                problem_gen_val,
                self.env.features,
                self.env.normalize,
                self.env.sort_func,
                self.env.reform,
            )
            eval_env = Monitor(eval_env)
            callback = EvalCallback(eval_env, **self.learn_params["eval_callback_kwargs"])
        else:
            callback = None

        tb_log_name = get_now() + "_" + self.model.__class__.__name__
        self.model.learn(total_timesteps, callback=callback, tb_log_name=tb_log_name)

    def reset(self):
        self.model.policy.apply(reset_weights)

    def _print_model(self):
        model_str = (
            f"{self.model.__class__.__name__}\n" f"```\n" f"{str(self.model.policy)}\n" f"```"
        )
        if self.model.logger is not None:
            model_str += f"\n- TB log: `{self.model.logger.dir}`"

        return model_str

    def save(self, save_path):
        save_path = Path(save_path)

        self.model.save(save_path)

        env_path = save_path.parent / f"{save_path.stem}.env"
        with env_path.open(mode="wb") as fid:
            dill.dump(self.env, fid)

    @classmethod
    def load(cls, load_path, model_cls=None, env=None, **kwargs):
        load_path = Path(load_path)

        if model_cls is None:
            cls_str = load_path.stem.split("_")[0]
            model_cls = cls.model_defaults[cls_str].cls
        elif isinstance(model_cls, str):
            model_cls = cls.model_defaults[model_cls].cls
        model = model_cls.load(load_path)

        if env is None:
            env_path = load_path.parent / f"{load_path.stem}.env"
            with env_path.open(mode="rb") as fid:
                env = dill.load(fid)

        return cls(env, model, **kwargs)


class MultiExtractor(BaseFeaturesExtractor):
    """
    Multiple-input feature extractor with valid action enforcement.

    Parameters
    ----------
    observation_space : gym.spaces.Space
    net_ch : nn.Module
    net_tasks: nn.Module

    """

    def __init__(self, observation_space: spaces.Dict, net_ch, net_tasks):
        super().__init__(observation_space, features_dim=1)  # `features_dim` must be overridden

        self.net_ch = net_ch
        self.net_tasks = net_tasks

        # Determine `features_dim` with single forward pass
        sample = observation_space.sample()
        # workaround SB3 encoding
        sample["seq"] = np.stack((sample["seq"], 1 - sample["seq"])).flatten(order="F")
        sample = {key: th.tensor(sample[key]).float().unsqueeze(0) for key in sample}
        with th.no_grad():
            self._features_dim = self.forward(sample).shape[1]  # SB3's workaround

    def forward(self, observations: dict):
        c, s, t = observations.values()
        t = t.permute(0, 2, 1)
        # s = s[:, ::2]  # override SB3 one-hot encoding
        # t = th.cat((t.permute(0, 2, 1), s.unsqueeze(1)), dim=1)
        # # reshape task features, combine w/ sequence mask

        c = self.net_ch(c)
        t = self.net_tasks(t)

        return th.cat((c, t), dim=-1)

    @classmethod
    def mlp(cls, observation_space, hidden_sizes_ch=(), hidden_sizes_tasks=()):
        n_ch = observation_space["ch_avail"].shape[-1]
        n_tasks, n_features = observation_space["tasks"].shape[-2:]

        layer_sizes_ch = [n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, last_act=True)

        layer_sizes_tasks = [n_tasks * n_features, *hidden_sizes_tasks]
        # layer_sizes_tasks = [n_tasks * (1 + n_features), *hidden_sizes_tasks]
        net_tasks = nn.Sequential(nn.Flatten(), *build_mlp(layer_sizes_tasks, last_act=True))

        return cls(observation_space, net_ch, net_tasks)

    @classmethod
    def cnn(
        cls,
        observation_space,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        kernel_sizes=2,
        cnn_kwargs=None,
    ):
        n_ch = observation_space["ch_avail"].shape[-1]
        n_features = observation_space["tasks"].shape[-1]

        layer_sizes_ch = [n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, last_act=True)

        layer_sizes_tasks = [n_features, *hidden_sizes_tasks]
        # layer_sizes_tasks = [1 + n_features, *hidden_sizes_tasks]
        if cnn_kwargs is None:
            cnn_kwargs = {}
        net_tasks = nn.Sequential(
            *build_cnn(layer_sizes_tasks, kernel_sizes, last_act=True, **cnn_kwargs),
            nn.Flatten(),
        )

        return cls(observation_space, net_ch, net_tasks)


class ValidActorCriticPolicy(ActorCriticPolicy):
    """Custom AC policy with valid action enforcement."""

    def __init__(self, *args, infer_valid_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        if callable(infer_valid_mask):
            self.infer_valid_mask = infer_valid_mask
        else:
            self.infer_valid_mask = lambda obs: np.zeros(self.observation_space.shape)

    def _get_action_dist_from_latent_valid(self, obs, latent_pi):  # added `obs` to signature
        mean_actions = self.action_net(latent_pi)
        # mask out invalid actions
        mean_actions = valid_logits(mean_actions, self.infer_valid_mask(obs))
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_valid(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent_valid(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_valid(obs, latent_pi)


class ValidQNetwork(QNetwork):
    def __init__(self, *args, infer_valid_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_valid_mask = infer_valid_mask

    def forward(self, obs):
        logits = self.q_net(self.extract_features(obs))
        return valid_logits(logits, self.infer_valid_mask(obs))


class ValidDQNPolicy(DQNPolicy):
    def __init__(self, *args, infer_valid_mask=None, **kwargs):
        if callable(infer_valid_mask):
            self.infer_valid_mask = infer_valid_mask
        else:
            self.infer_valid_mask = lambda obs: np.zeros(self.observation_space.shape)

        super().__init__(*args, **kwargs)

    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ValidQNetwork(**net_args, infer_valid_mask=self.infer_valid_mask).to(self.device)

import math
# from pathlib import Path
from abc import abstractmethod
from copy import deepcopy
from functools import partial, wraps
# from types import MethodType
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader

from task_scheduling.learning.environments import StepTasking
# from task_scheduling.learning.base import Base as BaseLearningScheduler
from task_scheduling.learning.supervised.base import Base as BaseSupervisedScheduler


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

NUM_WORKERS = 0
# NUM_WORKERS = os.cpu_count()

PIN_MEMORY = True
# PIN_MEMORY = False


def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class Base(BaseSupervisedScheduler):
    _learn_params_default = {
        'batch_size_train': 1,
        'n_gen_val': 0,
        'batch_size_val': 1,
        'weight_func': None,
        'max_epochs': 1,
        'shuffle': False,
    }

    def __init__(self, env, model, learn_params=None, valid_fwd=True):
        """
        Base class for PyTorch-based schedulers.

        Parameters
        ----------
        env : BaseTasking
            OpenAi gym environment.
        model : torch.nn.Module
            The learning network.
        learn_params : dict, optional
            Parameters used by the `learn` method.
        valid_fwd : bool, optional
            Enables wrapping of PyTorch module `forward` method with a parallel function that infers the valid action
            space and modifies the softmax output accordingly. Only relevant if the `env` type is `StepTasking`.

        """
        if not isinstance(model, nn.Module):
            raise TypeError("Argument `model` must be a `torch.nn.Module` instance.")

        if valid_fwd and (not isinstance(env, StepTasking) or env.seq_encoding is None):
            warn("Valid network can only be enforced using `StepTasking` environment with sequence encoding.")
            self.valid_fwd = False
        else:
            self.valid_fwd = valid_fwd

        if self.valid_fwd:
            def valid_wrapper(func):
                @wraps(func)
                def valid_func(*args, **kwargs):
                    p = func(*args, **kwargs)  # assumes softmax output for valid probabilities

                    mask = 1 - env.make_mask(*args, **kwargs)
                    p_mask = p * mask

                    idx_zero = p_mask.sum(dim=1) == 0.
                    p_mask[idx_zero] = mask[idx_zero]  # if no valid actions are non-zero, make them uniform

                    p_norm = functional.normalize(p_mask, p=1, dim=1)
                    return p_norm

                return valid_func

            model.forward = valid_wrapper(model.forward)  # FIXME: no longer a bound method!
            # model.forward = MethodType(valid_wrapper(model.forward), model)

        super().__init__(env, model, learn_params)

    def _process_obs(self, obs, normalize=False):
        """
        Estimate action probabilities given an observation.

        Parameters
        ----------
        obs : array_like
            Observation.
        normalize : bool, optional
            Enable normalization of model outputs.

        Returns
        -------
        numpy.ndarray
            Action probabilities.

        """
        _batch = True
        if obs.shape == self.env.observation_space.shape:
            _batch = False
            obs = obs[np.newaxis]
        else:
            raise NotImplementedError("Batch prediction not supported.")
        obs = obs.astype('float32')

        with torch.no_grad():
            # input_ = torch.from_numpy(obs[np.newaxis]).float()
            input_ = torch.from_numpy(obs)
            # input_ = input_.to(device)
            out = self.model(input_)

        if normalize:
            # out = functional.normalize(out, p=1, dim=-1)
            out = functional.softmax(out, dim=-1)

        out = out.numpy()
        if not _batch:
            out = out.squeeze(axis=0)

        return out

    def predict_prob(self, obs):
        return self._process_obs(obs, normalize=True)

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
        # return self._process_obs(obs).argmax()

        if self.valid_fwd:
            return self._process_obs(obs).argmax()
        else:
            # TODO: deprecate?
            p = self.predict_prob(obs)
            action = p.argmax()
            if action not in self.env.action_space:  # mask out invalid actions
                p = self.env.mask_probability(p)
                action = p.argmax()
            return action

    def reset(self):
        """Reset the learner."""
        self.model.apply(reset_weights)

    @abstractmethod
    def _fit(self, dl_train, dl_val, verbose=0):
        """
        Fit the PyTorch network.

        Parameters
        ----------
        dl_train : torch.utils.data.DataLoader
        dl_val : torch.utils.data.DataLoader
        verbose : {0, 1}, optional
            Enables progress print-out. 0: silent, 1: progress

        """
        raise NotImplementedError

    def learn(self, n_gen_learn, verbose=0):
        """
        Learn from the environment.

        Parameters
        ----------
        n_gen_learn : int
            Number of problems to generate data from.
        verbose : {0, 1, 2}, optional
            Progress print-out level. 0: silent, 1: add batch info, 2: add problem info

        """
        n_gen_val = self.learn_params['n_gen_val']
        if isinstance(n_gen_val, float) and n_gen_val < 1:  # convert fraction to number of problems
            n_gen_val = math.floor(n_gen_learn * n_gen_val)

        n_gen_train = n_gen_learn - n_gen_val

        if verbose >= 1:
            print("Generating training data...")
        x_train, y_train, *__ = self.env.data_gen_full(n_gen_train, weight_func=self.learn_params['weight_func'],
                                                       verbose=verbose)

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val, *__ = self.env.data_gen_full(n_gen_val, weight_func=self.learn_params['weight_func'],
                                                   verbose=verbose)

        # x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
        x_train, x_val = map(partial(torch.tensor, dtype=torch.float32), (x_train, x_val))
        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        ds_train = TensorDataset(x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=self.learn_params['batch_size_train'] * self.env.steps_per_episode,
                              shuffle=self.learn_params['shuffle'], pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=self.learn_params['batch_size_val'] * self.env.steps_per_episode,
                            shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        self._fit(dl_train, dl_val, verbose)


def build_torch_mlp(layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None):
    layers = []
    if start_layer is not None:
        layers.append(start_layer)
    for in_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(*in_out))
        layers.append(activation)
    layers.pop()
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


class TorchScheduler(Base):
    def __init__(self, env, model, loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None,
                 learn_params=None, valid_fwd=True):
        """
        Base class for pure PyTorch-based schedulers.

        Parameters
        ----------
        env : BaseTasking
            OpenAi gym environment.
        model : torch.nn.Module
            The PyTorch network.
        loss_func : callable, optional
        optim_cls : class, optional
            Optimizer class from `torch.nn.optim`.
        optim_params : dict, optional
            Arguments for optimizer instantiation.
        learn_params : dict, optional
            Parameters used by the `learn` method.
        valid_fwd : bool, optional
            Enables wrapping of PyTorch module `forward` method with a parallel function that infers the valid action
            space and modifies the softmax output accordingly. Only relevant if the `env` type is `StepTasking`.

        """
        super().__init__(env, model, learn_params, valid_fwd)

        # self.model = model.to(device)
        self.loss_func = loss_func
        # self.optimizer = optimizer
        if optim_params is None:
            optim_params = {}
        self.optimizer = optim_cls(self.model.parameters(), **optim_params)

    @classmethod
    def mlp(cls, env, hidden_layer_sizes=(), mlp_kwargs=None, loss_func=functional.cross_entropy, optim_cls=optim.Adam,
            optim_params=None, learn_params=None, valid_fwd=True):
        layer_sizes = [np.prod(env.observation_space.shape).item(), *hidden_layer_sizes, env.action_space.n]
        if mlp_kwargs is None:
            mlp_kwargs = {}
        if valid_fwd:
            mlp_kwargs['end_layer'] = nn.Softmax(dim=1)  # required for probability masking
        model = build_torch_mlp(layer_sizes, **mlp_kwargs)

        return cls(env, model, loss_func, optim_cls, optim_params, learn_params, valid_fwd)

    @classmethod
    def from_gen_mlp(cls, problem_gen, env_cls=StepTasking, env_params=None, hidden_layer_sizes=(), mlp_kwargs=None,
                     loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None, learn_params=None,
                     valid_fwd=True):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(env, hidden_layer_sizes, mlp_kwargs, loss_func, optim_cls, optim_params, learn_params,
                       valid_fwd)

    # @classmethod  # TODO: delete?
    # def from_env_mlp(cls, problem_gen, env_cls=StepTasking, env_params=None, hidden_layer_sizes=(), mlp_kwargs=None,
    #                  loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None, learn_params=None,
    #                  valid_fwd=True):
    #     if env_params is None:
    #         env_params = {}
    #     env = env_cls(problem_gen, **env_params)
    #
    #     layer_sizes = [np.prod(env.observation_space.shape).item(), *hidden_layer_sizes, env.action_space.n]
    #     if mlp_kwargs is None:
    #         mlp_kwargs = {}
    #     if valid_fwd:
    #         mlp_kwargs['end_layer'] = nn.Softmax(dim=1)  # required for probability masking
    #     model = build_torch_mlp(layer_sizes, **mlp_kwargs)
    #
    #     return cls(env, model, loss_func, optim_cls, optim_params, learn_params, valid_fwd)

    def _fit(self, dl_train, dl_val, verbose=0):
        if verbose >= 1:
            print('Training model...')

        def loss_batch(model, loss_func, xb_, yb_, opt=None):
            xb_, yb_ = xb_.to(device), yb_.to(device, dtype=torch.int64)
            loss = loss_func(model(xb_), yb_)

            if opt is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()

            return loss.item(), len(xb)

        for epoch in range(self.learn_params['max_epochs']):
            self.model.train()
            for xb, yb in dl_train:
                loss_batch(self.model, self.loss_func, xb, yb, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(self.model, self.loss_func, xb, yb) for xb, yb in dl_val]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            if verbose >= 1:
                print(f"  Epoch = {epoch} : loss = {val_loss:.3f}", end='\r')

    def learn(self, n_gen_learn, verbose=0):
        self.model = self.model.to(device)
        super().learn(n_gen_learn, verbose)
        self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`

    # def save(self, save_path=None):
    #     if save_path is None:
    #         save_path = f"models/temp/{NOW_STR}.pth"
    #
    #     with Path(save_path).joinpath('env').open(mode='wb') as fid:
    #         dill.dump(self.env, fid)  # save environment
    #
    #     torch.save(self.model, save_path)
    #
    # @classmethod
    # def load(cls, load_path):
    #     model = torch.load(load_path)
    #
    #     with Path(load_path).joinpath('env').open(mode='rb') as fid:
    #         env = dill.load(fid)
    #
    #     return cls(model, env)


class LitModel(pl.LightningModule):
    def __init__(self, module, loss_func=functional.cross_entropy, optim_cls=torch.optim.Adam, optim_params=None):
        super().__init__()

        self.module = module
        self.loss_func = loss_func
        self.optim_cls = optim_cls
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

    @classmethod
    def mlp(cls, layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None,
            loss_func=functional.cross_entropy, optim_cls=torch.optim.Adam, optim_params=None):
        module = build_torch_mlp(layer_sizes, activation, start_layer, end_layer)
        return cls(module, loss_func, optim_cls, optim_params)

    def forward(self, x):
        return self.module(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optim_cls(self.parameters(), **self.optim_params)


class LitScheduler(Base):
    def __init__(self, env, model, trainer_kwargs=None, learn_params=None, valid_fwd=True):
        """
        Base class for PyTorch Lightning-based schedulers.

        Parameters
        ----------
        env : BaseTasking
            OpenAi gym environment.
        model : torch.nn.Module
            The PyTorch-Lightning network.
        trainer_kwargs : dict, optional
            Arguments passed to instantiation of pl.Trainer object.
        learn_params : dict, optional
            Parameters used by the `learn` method.
        valid_fwd : bool, optional
            Enables wrapping of PyTorch module `forward` method with a parallel function that infers the valid action
            space and modifies the softmax output accordingly. Only relevant if the `env` type is `StepTasking`.

        """
        super().__init__(env, model, learn_params, valid_fwd)

        if trainer_kwargs is None:
            trainer_kwargs = {}
        self.trainer_kwargs = trainer_kwargs

        # Note: the kwargs below are specified in `learn_params` for consistency with `TorchScheduler`
        self.trainer_kwargs.update({
            'max_epochs': self.learn_params['max_epochs'],
        })
        self.trainer = pl.Trainer(**self.trainer_kwargs)

    @classmethod
    def from_module(cls, env, module, model_kwargs=None, trainer_kwargs=None, learn_params=None, valid_fwd=True):
        if model_kwargs is None:
            model_kwargs = {}
        model = LitModel(module, **model_kwargs)
        return cls(env, model, trainer_kwargs, learn_params, valid_fwd)

    @classmethod
    def mlp(cls, env, hidden_layer_sizes, mlp_kwargs=None, trainer_kwargs=None, learn_params=None, valid_fwd=True):
        layer_sizes = [np.prod(env.observation_space.shape).item(), *hidden_layer_sizes, env.action_space.n]
        if mlp_kwargs is None:
            mlp_kwargs = {}
        if valid_fwd:
            mlp_kwargs['end_layer'] = nn.Softmax(dim=1)  # required for probability masking
        model = LitModel.mlp(layer_sizes, **mlp_kwargs)

        return cls(env, model, trainer_kwargs, learn_params, valid_fwd)

    @classmethod
    def from_gen_mlp(cls, problem_gen, env_cls=StepTasking, env_params=None, hidden_layer_sizes=(), mlp_kwargs=None,
                     trainer_kwargs=None, learn_params=None, valid_fwd=True):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(env, hidden_layer_sizes, mlp_kwargs, trainer_kwargs, learn_params, valid_fwd)

    def reset(self):
        super().reset()
        self.trainer = pl.Trainer(**deepcopy(self.trainer_kwargs))

    def _fit(self, dl_train, dl_val, verbose=0):
        if verbose >= 1:
            print('Training model...')

        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.progress.ProgressBar):
                cb._refresh_rate = int(verbose >= 1)

        self.trainer.fit(self.model, dl_train, dl_val)

import math
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader

from task_scheduling.mdp.environments import Base as BaseEnv, Index
from task_scheduling.mdp.supervised.base import Base as BaseSupervisedScheduler


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

    def __init__(self, env, model, learn_params=None):
        """
        Base class for PyTorch-based schedulers.

        Parameters
        ----------
        env : BaseEnv
            OpenAi gym environment.
        model : torch.nn.Module
            The learning network.
        learn_params : dict, optional
            Parameters used by the `learn` method.

        """
        if not isinstance(model, nn.Module):
            raise TypeError("Argument `model` must be a `torch.nn.Module` instance.")

        super().__init__(env, model, learn_params)

    @classmethod
    def from_gen(cls, problem_gen, env_cls=Index, env_params=None, *args, **kwargs):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)
        return cls(env, *args, **kwargs)

    @staticmethod
    def _obs_to_tuple(obs):
        if obs.dtype.names is not None:
            return tuple(obs[key] for key in obs.dtype.names)
        else:
            return obs,

    def _process_obs(self, obs, softmax=False):
        """
        Estimate action probabilities given an observation.

        Parameters
        ----------
        obs : array_like
            Observation.
        softmax : bool, optional
            Enable normalization of model outputs.

        Returns
        -------
        numpy.ndarray
            Action probabilities.

        """

        # input_ = (torch.from_numpy(o[np.newaxis]).float() for o in self._obs_to_tuple(obs))
        input_ = (torch.from_numpy(o).float().unsqueeze(0) for o in self._obs_to_tuple(obs))
        # input_ = input_.to(device)
        with torch.no_grad():
            out = self.model(*input_)

        if softmax:
            out = functional.softmax(out, dim=-1)

        out = out.numpy()
        out = out.squeeze(axis=0)

        return out

    def predict_prob(self, obs):
        return self._process_obs(obs, softmax=True)

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

        # TODO: deprecate?
        if not hasattr(self.model, 'valid_fwd') or self.model.valid_fwd:
            return self._process_obs(obs).argmax()
        else:
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

        x_train = tuple(map(partial(torch.tensor, dtype=torch.float32), self._obs_to_tuple(x_train)))
        x_val = tuple(map(partial(torch.tensor, dtype=torch.float32), self._obs_to_tuple(x_val)))

        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        ds_train = TensorDataset(*x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=self.learn_params['batch_size_train'] * self.env.steps_per_episode,
                              shuffle=self.learn_params['shuffle'], pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        ds_val = TensorDataset(*x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=self.learn_params['batch_size_val'] * self.env.steps_per_episode,
                            shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        self._fit(dl_train, dl_val, verbose)

    # def save(self, save_path):
    #     save_path = Path(save_path)
    #
    #     if save_path.is_file():
    #         pass
    #     # file.parent.mkdir(parents=True, exist_ok=True)
    #
    #     with Path(save_path).joinpath('env').open(mode='wb') as fid:
    #         dill.dump(self.env, fid)  # save environment
    #
    #     torch.save(self.model, save_path)
    #
    # @classmethod
    # def load(cls, load_path, *args, **kwargs):
    #     model = torch.load(load_path)
    #
    #     with Path(load_path).joinpath('env').open(mode='rb') as fid:
    #         env = dill.load(fid)
    #
    #     return cls(env, model, *args, **kwargs)


def _build_mlp(layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None):
    """
    PyTorch-Lightning sequential MLP.

    Parameters
    ----------
    layer_sizes : iterable of int
        Hidden layer sizes.
    activation : nn.Module, optional
    start_layer : nn.Module, optional
    end_layer : nn.Module, optional

    Returns
    -------
    nn.Sequential

    """
    layers = []
    if start_layer is not None:
        layers.append(start_layer)
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if i < len(layer_sizes) - 2:
            layers.append(activation)
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


class ValidNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, ch_avail, seq, tasks):
        y = self.module(ch_avail, tasks)
        y = y - 1e6 * seq  # TODO: try different masking operations?
        return y


class MultiMLP(nn.Module):
    def __init__(self, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=()):
        super().__init__()

        size_in_ch = np.prod(env.observation_space['ch_avail'].shape).item()
        layer_sizes_ch = [size_in_ch, *hidden_sizes_ch]
        end_layer_ch = nn.ReLU() if bool(hidden_sizes_ch) else None
        self.mlp_ch = _build_mlp(layer_sizes_ch, end_layer=end_layer_ch)

        size_in_tasks = np.prod(env.observation_space['tasks'].shape).item()
        layer_sizes_tasks = [size_in_tasks, *hidden_sizes_tasks]
        end_layer_tasks = nn.ReLU() if bool(hidden_sizes_tasks) else None
        self.mlp_tasks = _build_mlp(layer_sizes_tasks, end_layer=end_layer_tasks)

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        self.mlp_joint = _build_mlp(layer_sizes_joint, start_layer=None)

    def forward(self, ch_avail, tasks):
        c = self.mlp_ch(ch_avail)
        t = self.mlp_tasks(tasks)

        x = torch.cat((c, t), dim=-1)
        x = self.mlp_joint(x)
        return x


class VaryCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        n_filter = 400
        l_kernel = 2

        # TODO: padding mode?
        # self.conv1 = nn.Conv2d(1, n_filter, kernel_size=(l_kernel, n_features), padding=(l_kernel-1, 0))
        # self.conv2 = nn.Conv1d(n_filter, 1, kernel_size=(l_kernel,), padding=l_kernel - 1)
        self.conv1 = nn.Conv2d(1, n_filter, kernel_size=(l_kernel, n_features))
        self.conv2 = nn.Conv1d(n_filter, 1, kernel_size=(l_kernel,))

    def forward(self, ch_avail, tasks):  # TODO: chan info?
        x = tasks

        n_batch, n_tasks, n_features = x.shape
        device_ = x.device

        x = x.view(n_batch, 1, n_tasks, n_features)

        pad = torch.zeros(n_batch, 1, self.conv1.kernel_size[0] - 1, n_features, device=device_)
        x = torch.cat((x, pad), dim=2)
        x = self.conv1(x)
        x = x.squeeze(dim=3)
        x = functional.relu(x)

        pad = torch.zeros(n_batch, self.conv2.in_channels, self.conv2.kernel_size[0] - 1, device=device_)
        x = torch.cat((x, pad), dim=2)
        x = self.conv2(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        return x


class TorchScheduler(Base):
    def __init__(self, env, model, loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None,
                 learn_params=None):
        """
        Base class for pure PyTorch-based schedulers.

        Parameters
        ----------
        env : BaseEnv
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

        """
        super().__init__(env, model, learn_params)

        # self.model = model.to(device)
        self.loss_func = loss_func
        if optim_params is None:
            optim_params = {}
        self.optimizer = optim_cls(self.model.parameters(), **optim_params)

    @classmethod
    def mlp(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=(),
            loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None, learn_params=None):
        model = MultiMLP(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint)
        model = ValidNet(model)
        return cls(env, model, loss_func, optim_cls, optim_params, learn_params)

    @classmethod
    def from_gen_mlp(cls, problem_gen, env_cls=Index, env_params=None, hidden_sizes_ch=(), hidden_sizes_tasks=(),
                     hidden_sizes_joint=(), loss_func=functional.cross_entropy, optim_cls=optim.Adam, optim_params=None,
                     learn_params=None):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint, loss_func, optim_cls, optim_params,
                       learn_params)

    def _fit(self, dl_train, dl_val, verbose=0):
        if verbose >= 1:
            print('Training model...')

        def loss_batch(model, loss_func, batch_, opt=None):
            batch_ = [t.to(device) for t in batch_]
            xb_, yb_ = batch_[:-1], batch_[-1]
            loss = loss_func(model(*xb_), yb_)

            if opt is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()

            return loss.item(), len(xb_)

        for epoch in range(self.learn_params['max_epochs']):
            self.model.train()
            for batch in dl_train:
                loss_batch(self.model, self.loss_func, batch, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(self.model, self.loss_func, batch) for batch in dl_val]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            if verbose >= 1:
                print(f"  Epoch = {epoch} : loss = {val_loss:.3f}", end='\r')

    def learn(self, n_gen_learn, verbose=0):
        self.model = self.model.to(device)
        super().learn(n_gen_learn, verbose)
        self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`


class LitModel(pl.LightningModule):
    def __init__(self, module, loss_func=functional.cross_entropy, optim_cls=torch.optim.Adam, optim_params=None):
        super().__init__()

        self.module = module
        self.loss_func = loss_func
        self.optim_cls = optim_cls
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1]
        y_hat = self(*x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1]
        y_hat = self(*x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optim_cls(self.parameters(), **self.optim_params)


class LitScheduler(Base):
    def __init__(self, env, model, trainer_kwargs=None, learn_params=None):
        """
        Base class for PyTorch Lightning-based schedulers.

        Parameters
        ----------
        env : BaseEnv
            OpenAi gym environment.
        model : torch.nn.Module
            The PyTorch-Lightning network.
        trainer_kwargs : dict, optional
            Arguments passed to instantiation of pl.Trainer object.
        learn_params : dict, optional
            Parameters used by the `learn` method.

        """
        super().__init__(env, model, learn_params)

        if trainer_kwargs is None:
            trainer_kwargs = {}
        self.trainer_kwargs = trainer_kwargs

        # Note: the kwargs below are specified in `learn_params` for consistency with `TorchScheduler`
        self.trainer_kwargs.update({
            'max_epochs': self.learn_params['max_epochs'],
        })
        self.trainer = pl.Trainer(**self.trainer_kwargs)

    @classmethod
    def from_module(cls, env, module, model_kwargs=None, trainer_kwargs=None, learn_params=None):
        if model_kwargs is None:
            model_kwargs = {}
        model = LitModel(module, **model_kwargs)
        return cls(env, model, trainer_kwargs, learn_params)

    @classmethod
    def from_gen_module(cls, problem_gen, module, env_cls=Index, env_params=None, model_kwargs=None,
                        trainer_kwargs=None, learn_params=None):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        cls.from_module(env, module, model_kwargs, trainer_kwargs, learn_params)

    @classmethod
    def mlp(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=(), model_kwargs=None,
            trainer_kwargs=None, learn_params=None):
        module = MultiMLP(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint)
        module = ValidNet(module)
        return cls.from_module(env, module, model_kwargs, trainer_kwargs, learn_params)

    @classmethod
    def from_gen_mlp(cls, problem_gen, env_cls=Index, env_params=None, hidden_sizes_ch=(), hidden_sizes_tasks=(),
                     hidden_sizes_joint=(), model_kwargs=None, trainer_kwargs=None, learn_params=None):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint, model_kwargs, trainer_kwargs,
                       learn_params)

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

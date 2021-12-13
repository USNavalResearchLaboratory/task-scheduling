import math
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from inspect import signature
from pathlib import Path

import dill
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

num_workers = 0
# num_workers = os.cpu_count()

pin_memory = True
# pin_memory = False


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
        x_train, y_train, *w_train = self.env.data_gen_full(n_gen_train, weight_func=self.learn_params['weight_func'],
                                                            verbose=verbose)

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val, *w_val = self.env.data_gen_full(n_gen_val, weight_func=self.learn_params['weight_func'],
                                                      verbose=verbose)

        x_train = tuple(map(partial(torch.tensor, dtype=torch.float32), self._obs_to_tuple(x_train)))
        x_val = tuple(map(partial(torch.tensor, dtype=torch.float32), self._obs_to_tuple(x_val)))

        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        tensors_train = [*x_train, y_train]
        tensors_val = [*x_val, y_val]

        if callable(self.learn_params['weight_func']):
            w_train, w_val = map(partial(torch.tensor, dtype=torch.float32), (w_train[0], w_val[0]))
            tensors_train.append(w_train)
            tensors_val.append(w_val)

        # ds_train = TensorDataset(*x_train, y_train)
        ds_train = TensorDataset(*tensors_train)
        dl_train = DataLoader(ds_train, batch_size=self.learn_params['batch_size_train'] * self.env.steps_per_episode,
                              shuffle=self.learn_params['shuffle'], pin_memory=pin_memory, num_workers=num_workers)

        ds_val = TensorDataset(*tensors_val)
        dl_val = DataLoader(ds_val, batch_size=self.learn_params['batch_size_val'] * self.env.steps_per_episode,
                            shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

        self._fit(dl_train, dl_val, verbose)

    def save(self, save_path):
        torch.save(self.model, save_path)

        save_path = Path(save_path)
        env_path = save_path.parent / f'{save_path.stem}.env'

        with env_path.open(mode='wb') as fid:
            dill.dump(self.env, fid)  # save environment

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        model = torch.load(load_path)

        if env is None:
            load_path = Path(load_path)
            env_path = load_path.parent / f'{load_path.stem}.env'
            with env_path.open(mode='rb') as fid:
                env = dill.load(fid)

        return cls(env, model, **kwargs)


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


def valid_logits(x, seq):
    return x - 1e6 * seq  # TODO: try different masking operations?


# def valid_wrapper(func):  # TODO: make `wraps` work
#     # @wraps(func)
#     def valid_fwd(self, ch_avail, seq, tasks):
#         y = func(self, ch_avail, tasks)
#         y = valid_logits(y, seq)
#         return y
#     return valid_fwd


class UniMLP(nn.Module):
    def __init__(self, env, hidden_sizes=()):
        super().__init__()

        layer_sizes = [
            np.prod(env.observation_space.shape).item(),
            *hidden_sizes,
            env.action_space.n,
        ]
        self.mlp = _build_mlp(layer_sizes)

    def forward(self, x):
        return self.mlp(x)


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

    def forward(self, ch_avail, seq, tasks):
        c = self.mlp_ch(ch_avail)
        t = self.mlp_tasks(tasks)

        x = torch.cat((c, t), dim=-1)
        x = self.mlp_joint(x)

        x = valid_logits(x, seq)
        return x


class VaryCNN(nn.Module):
    def __init__(self, env, kernel_len):
        super().__init__()

        self.n_features = len(env.features)
        # self.n_ch = env.n_ch

        self.n_filter = 400

        self.conv2d = nn.Conv2d(1, self.n_filter, kernel_size=(kernel_len, self.n_features))
        self.conv1 = nn.Conv1d(self.n_filter, 1, kernel_size=kernel_len)
        # self.conv1 = nn.Conv1d(self.n_filter + self.n_ch, 1, kernel_size=(l_kernel,))

        # self.affine_ch = nn.Linear(self.n_ch, self.n_filter, bias=False)
        self.conv_ch = nn.Conv1d(1, self.n_filter, kernel_size=(3,), padding='same')

    def forward(self, ch_avail, seq, tasks):
        c, t = ch_avail, tasks

        n_batch, __, n_tasks, n_features = t.shape
        device_ = t.device

        # t = t.unsqueeze(dim=1)

        pad = torch.zeros(n_batch, 1, self.conv2d.kernel_size[0] - 1, n_features, device=device_)
        t = torch.cat((t, pad), dim=2)
        t = self.conv2d(t)
        t = t.squeeze(dim=3)

        # c = self.affine_ch(c)
        # c = c.unsqueeze(-1)

        c = self.conv_ch(c.unsqueeze(1))
        c = functional.adaptive_avg_pool1d(c, (1,))

        x = t + c
        x = functional.relu(x)
        # c = c.unsqueeze(-1).expand(n_batch, self.n_ch, n_tasks)
        # x = torch.cat((t, z), dim=1)

        pad = torch.zeros(n_batch, self.conv1.in_channels, self.conv1.kernel_size[0] - 1, device=device_)
        x = torch.cat((x, pad), dim=2)
        x = self.conv1(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        x = valid_logits(x, seq)
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

            if callable(self.learn_params['weight_func']):
                xb_, yb_, wb_ = batch_[:-2], batch_[-2], batch_[-1]
                losses_ = loss_func(model(*xb_), yb_, reduction='none')
                loss = torch.mean(wb_ * losses_)
            else:
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

        _sig_fwd = signature(module.forward)
        self._n_in = len(_sig_fwd.parameters)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _process_batch(self, batch, _batch_idx):
        if len(batch) > self._n_in + 1:  # includes sample weighting
            x, y, w = batch[:-2], batch[-2], batch[-1]
            losses = self.loss_func(self(*x), y, reduction='none')
            loss = torch.mean(w * losses)
        elif len(batch) == self._n_in + 1:
            x, y = batch[:-1], batch[-1]
            loss = self.loss_func(self(*x), y)
        else:
            raise ValueError

        # x, y = batch[:-1], batch[-1]
        # y_hat = self(*x)
        # loss = self.loss_func(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx)
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

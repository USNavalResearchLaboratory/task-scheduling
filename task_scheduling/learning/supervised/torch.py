from functools import partial
from pathlib import Path
from abc import abstractmethod
import math
from copy import deepcopy

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl

from task_scheduling.learning.base import Base as BaseLearningScheduler


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

NUM_WORKERS = 0
# NUM_WORKERS = os.cpu_count()

AVAIL_GPUS = min(1, torch.cuda.device_count())

PIN_MEMORY = True
# PIN_MEMORY = False


def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class Base(BaseLearningScheduler):
    _learn_params_default = {
        'batch_size_train': 1,
        'n_gen_val': 0,
        'batch_size_val': 1,
        'weight_func': None,
        'max_epochs': 1,
        'shuffle': False,
        'callbacks': []
    }

    def predict_prob(self, obs):
        _batch = True
        if obs.shape == self.env.observation_space.shape:
            _batch = False
            obs = obs[np.newaxis]
        else:
            raise NotImplementedError("Batch prediction not supported.")
        obs = obs.astype('float32')

        with torch.no_grad():
            # input_ = torch.from_numpy(obs[np.newaxis]).float()  # TODO: tensor conversion in model?
            input_ = torch.from_numpy(obs)
            # input_ = input_.to(device)
            prob = self.model(input_)

        prob = prob.numpy()
        if not _batch:
            prob = prob.squeeze(axis=0)

        return prob

    def predict(self, obs):
        p = self.predict_prob(obs)
        action = p.argmax()

        if action not in self.env.action_space:  # mask out invalid actions
            p = self.env.mask_probability(p)
            action = p.argmax()

        return action

    def reset(self):
        self.model.apply(reset_weights)

    @abstractmethod
    def _fit(self, dl_train, dl_val, verbose=0):
        raise NotImplementedError

    def learn(self, n_gen_learn, verbose=0):

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

        # if callable(weight_func):  # FIXME: add sample weighting (validation, too)
        #     fit_params['sample_weight'] = d_train[2]

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=self.learn_params['batch_size_val'] * self.env.steps_per_episode,
                            shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        self._fit(dl_train, dl_val, verbose)

    # def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
    #           fit_params=None, verbose=0):
    #
    #     if verbose >= 1:
    #         print("Generating training data...")
    #     x_train, y_train, *__ = self.env.data_gen_full(n_batch_train * batch_size_train, weight_func=weight_func,
    #                                                    verbose=verbose)
    #
    #     if verbose >= 1:
    #         print("Generating validation data...")
    #     x_val, y_val, *__ = self.env.data_gen_full(n_batch_val * batch_size_val, weight_func=weight_func,
    #                                                verbose=verbose)
    #
    #     # x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
    #     x_train, x_val = map(partial(torch.tensor, dtype=torch.float32), (x_train, x_val))
    #     y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))
    #
    #     ds_train = TensorDataset(x_train, y_train)
    #     dl_train = DataLoader(ds_train, batch_size=batch_size_train * self.env.steps_per_episode,
    #                           shuffle=fit_params['shuffle'], pin_memory=PIN_MEMORY)
    #
    #     # if callable(weight_func):  # FIXME: add sample weighting (validation, too)
    #     #     fit_params['sample_weight'] = d_train[2]
    #
    #     ds_val = TensorDataset(x_val, y_val)
    #     dl_val = DataLoader(ds_val, batch_size=batch_size_val * self.env.steps_per_episode, shuffle=False,
    #                         pin_memory=PIN_MEMORY)
    #
    #     self._fit(dl_train, dl_val, fit_params, verbose)


class TorchScheduler(Base):
    log_dir = Path.cwd() / 'logs/learn'

    def __init__(self, env, model, loss_func, opt, learn_params=None):
        super().__init__(env, model, learn_params)

        # self.model = model.to(device)
        self.loss_func = loss_func
        self.opt = opt

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
                loss_batch(self.model, self.loss_func, xb, yb, self.opt)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(self.model, self.loss_func, xb, yb) for xb, yb in dl_val]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            if verbose >= 1:
                print(f"  Epoch = {epoch} : loss = {val_loss:.3f}", end='\r')

        # TODO: loss/acc plots, tensorboard, etc.

    def learn(self, n_gen_learn, verbose=0):
        self.model = self.model.to(device)
        super().learn(n_gen_learn, verbose)
        self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`

    # def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
    #           fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):
    #
    #     self.model = self.model.to(device)
    #     super().learn(n_batch_train, batch_size_train, n_batch_val, batch_size_val, weight_func, fit_params, verbose,
    #                   do_tensorboard, plot_history)
    #     self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`

    # def save(self, save_path=None):  # FIXME FIXME
    #     if save_path is None:
    #         save_path = f"models/temp/{NOW_STR}.pkl"
    #
    #     with Path(save_path).open(mode='wb') as fid:
    #         dill.dump(self, fid)
    #
    # @classmethod
    # def load(cls, load_path):
    #     with Path(load_path).open(mode='rb') as fid:
    #         scheduler = dill.load(fid)
    #
    #     return scheduler

    # def save(self, save_path=None):  # FIXME FIXME
    #     if save_path is None:
    #         save_path = f"models/temp/{NOW_STR}.pth"
    #
    #     with Path(save_path).joinpath('env').open(mode='wb') as fid:
    #         dill.dump(self.env, fid)  # save environment
    #
    #     torch.save(self.model, save_path)
    #     # self.model.save(save_path)  # save TF model  # FIXME
    #
    # @classmethod
    # def load(cls, load_path):
    #     model = torch.load(load_path)
    #     # model = keras.models.load_model(load_path)  # FIXME
    #
    #     with Path(load_path).joinpath('env').open(mode='rb') as fid:
    #         env = dill.load(fid)
    #
    #     return cls(model, env)  # FIXME: opt, loss, etc.?


class LitScheduler(Base):
    log_dir = Path.cwd() / 'logs/learn'

    def __init__(self, env, model, learn_params):
        super().__init__(env, model, learn_params)

        self.trainer_params = {
            'gpus': AVAIL_GPUS,
            # 'distributed_backend': 'ddp',
            # 'profiler': 'simple',
            'checkpoint_callback': False,
            'logger': True,
            'default_root_dir': str(self.log_dir),
            # 'progress_bar_refresh_rate': 0,
            'max_epochs': self.learn_params['max_epochs'],
            'callbacks': self.learn_params['callbacks'],
        }
        self.trainer = pl.Trainer(**self.trainer_params)

    def reset(self):  # TODO: add reset method to predictor base class?
        super().reset()
        self.trainer = pl.Trainer(**deepcopy(self.trainer_params))

    def _fit(self, dl_train, dl_val, verbose=0):

        # TODO: sample weighting?

        if verbose >= 1:
            print('Training model...')

        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.progress.ProgressBar):
                cb._refresh_rate = int(verbose >= 1)

        self.trainer.fit(self.model, dl_train, dl_val)

        # TODO: loss/acc plots, tensorboard, etc.

    # def save(self, save_path=None):  # FIXME FIXME
    #     if save_path is None:
    #         save_path = f"models/temp/{NOW_STR}.pth"
    #
    #     with Path(save_path).joinpath('env').open(mode='wb') as fid:
    #         dill.dump(self.env, fid)  # save environment
    #
    #     torch.save(self.model, save_path)
    #     # self.model.save(save_path)  # save TF model  # FIXME
    #
    # @classmethod
    # def load(cls, load_path):
    #     model = torch.load(load_path)
    #     # model = keras.models.load_model(load_path)  # FIXME
    #
    #     with Path(load_path).joinpath('env').open(mode='rb') as fid:
    #         env = dill.load(fid)
    #
    #     return cls(model, env)  # FIXME: opt, loss, etc.?

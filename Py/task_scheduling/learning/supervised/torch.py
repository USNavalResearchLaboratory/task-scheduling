import os
from functools import partial
from pathlib import Path
import dill
from abc import abstractmethod

import numpy as np

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from task_scheduling.learning.supervised.base import Base


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

AVAIL_CPUS = os.cpu_count()
AVAIL_GPUS = min(1, torch.cuda.device_count())

PIN_MEMORY = True
# PIN_MEMORY = False


def weights_init(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class BaseTorch(Base):
    def obs_to_prob(self, obs):
        with torch.no_grad():
            input_ = torch.from_numpy(obs[np.newaxis]).float()  # TODO: tensor conversion in model?
            # input_ = input_.to(device)
            prob = self.model(input_).squeeze(0)

        return prob

    def _print_model(self, file=None):
        print(self.model, file=file)

    def reset(self):
        self.model.apply(weights_init)

    @abstractmethod
    def _fit(self, dl_train, dl_val, fit_params=None, verbose=0):
        raise NotImplementedError

    def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        if verbose >= 1:
            print("Generating training data...")
        x_train, y_train, *__ = self.env.data_gen_numpy(n_batch_train * batch_size_train, weight_func=weight_func,
                                                        verbose=verbose)

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val, *__ = self.env.data_gen_numpy(n_batch_val * batch_size_val, weight_func=weight_func,
                                                    verbose=verbose)

        # x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
        x_train, x_val = map(partial(torch.tensor, dtype=torch.float32), (x_train, x_val))
        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        ds_train = TensorDataset(x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size_train * self.env.steps_per_episode,
                              shuffle=fit_params['shuffle'], pin_memory=PIN_MEMORY)

        # if callable(weight_func):  # FIXME: add sample weighting (validation, too)
        #     fit_params['sample_weight'] = d_train[2]

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=batch_size_val * self.env.steps_per_episode, shuffle=False,
                            pin_memory=PIN_MEMORY)

        self._fit(dl_train, dl_val, fit_params, verbose)


class TorchScheduler(BaseTorch):
    log_dir = Path.cwd() / 'logs' / 'torch_train'

    def __init__(self, env, model, loss_func, opt):
        super().__init__(env, model)

        # self.model = model.to(device)
        self.loss_func = loss_func
        self.opt = opt

    def _fit(self, dl_train, dl_val, fit_params=None, verbose=0):

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

        epochs = fit_params['epochs']
        for epoch in range(epochs):
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

    def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        self.model = self.model.to(device)
        super().learn(n_batch_train, batch_size_train, n_batch_val, batch_size_val, weight_func, fit_params, verbose,
                      do_tensorboard, plot_history)
        self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`

    # def save(self, save_path=None):  # FIXME FIXME
    #     if save_path is None:
    #         save_path = f"models/temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
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


class LitScheduler(BaseTorch):
    log_dir = Path.cwd() / 'logs'

    def __init__(self, env, model):
        super().__init__(env, model)

        # self.trainer = pl.Trainer(gpus=AVAIL_GPUS)

    def _fit(self, dl_train, dl_val, fit_params=None, verbose=0):

        # TODO: sample weighting?

        if verbose >= 1:
            print('Training model...')

        trainer_kwargs = {
            'gpus': AVAIL_GPUS,
            'logger': True,
            'default_root_dir': self.log_dir.as_posix(),
            # 'progress_bar_refresh_rate': 1000,
        }

        try:
            callbacks = fit_params['callbacks']  # FIXME: rework, remove
        except KeyError:
            callbacks = None

        trainer = pl.Trainer(max_epochs=fit_params['epochs'], callbacks=callbacks, **trainer_kwargs)
        trainer.fit(self.model, dl_train, dl_val)
        # self.trainer.fit(self.model, dl_train, dl_val)

        # TODO: loss/acc plots, tensorboard, etc.

    # def save(self, save_path=None):  # FIXME FIXME
    #     if save_path is None:
    #         save_path = f"models/temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
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

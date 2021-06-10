import shutil
import time
import webbrowser
from functools import partial
from pathlib import Path
import dill

import gym
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

# from tensorboard import program

# from task_scheduling.learning import environments as envs
from task_scheduling.learning.supervised.base import BaseSupervisedScheduler


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

AVAIL_GPUS = min(1, torch.cuda.device_count())


def weights_init(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class Scheduler(BaseSupervisedScheduler):
    log_dir = Path.cwd() / 'logs' / 'torch_train'

    def __init__(self, env, model, loss_func, opt):
        self.env = env
        # if not isinstance(self.env.action_space, gym.spaces.Discrete):
        #     raise TypeError("Action space must be Discrete.")

        self.model = model
        # self.model = model.to(device)
        self.loss_func = loss_func
        self.opt = opt

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Sequence of task_scheduling.tasks.Base
        ch_avail : Sequence of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        # ensure_valid = isinstance(self.env, envs.StepTasking) and not self.env.do_valid_actions
        # # ensure_valid = False    # TODO: trained models may naturally avoid invalid actions!!

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)

        done = False
        while not done:
            with torch.no_grad():
                input_ = torch.from_numpy(obs[np.newaxis]).float()  # TODO: tensor conversion in model?
                # input_ = input_.to(device)
                prob = self.model(input_).squeeze(0)
                # prob = self.model(input_).numpy().squeeze(0)
            # prob = np.zeros(self.env.action_space.n)

            try:
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)
            except ValueError:
                prob = self.env.mask_probability(prob)
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)

            # if ensure_valid:  # TODO: deprecate
            #     prob = self.env.mask_probability(prob)
            # action = prob.argmax()
            #
            # obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def summary(self, file=None):
        print('Env: ', end='', file=file)
        self.env.summary(file)
        print('Model\n---\n```', file=file)

        print(self.model, file=file)
        print('```', end='\n\n', file=file)

    def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        self.model = self.model.to(device)

        if verbose >= 1:
            print("Generating training data...")
        x_train, y_train, *__ = self.env.data_gen_numpy(n_batch_train * batch_size_train, weight_func=weight_func, verbose=verbose)
        # d_train = self.env.data_gen_numpy(n_batch_train * batch_size_train, weight_func=weight_func, verbose=verbose)
        # x_train, y_train = d_train[:2]

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val = self.env.data_gen_numpy(n_batch_val * batch_size_val, weight_func=weight_func, verbose=verbose)


        # x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
        x_train, x_val = map(partial(torch.tensor, dtype=torch.float32), (x_train, x_val))
        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        ds_train = TensorDataset(x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size_train * self.env.steps_per_episode, shuffle=True,
                              pin_memory=True)
        # FIXME: shuffle control? Enforce False??

        # if callable(weight_func):  # FIXME: add sample weighting
        #     fit_params['sample_weight'] = d_train[2]

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=batch_size_val * self.env.steps_per_episode, shuffle=True,
                            pin_memory=True)

        # TODO: validation weighting?

        # # Add stopping callback if needed
        # if 'callbacks' not in fit_params:
        #     fit_params['callbacks'] = [keras.callbacks.EarlyStopping('val_loss', patience=60, min_delta=0.)]
        # elif not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in fit_params['callbacks']):
        #     fit_params['callbacks'].append(keras.callbacks.EarlyStopping('val_loss', patience=60, min_delta=0.))

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

        # TODO: loss/acc plots, tensorboard, etc. LIGHTNING??

        self.model = self.model.to('cpu')  # move back to CPU for single sample evaluations in `__call__`

    def reset(self):
        self.model.apply(weights_init)

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
    #     return cls(model, env)  # FIXME: opt, loss, etc.? Easier with Lightning???


class LightningScheduler(BaseSupervisedScheduler):
    log_dir = Path.cwd() / 'logs' / 'pl_train'

    def __init__(self, env, model):
        self.env = env
        # if not isinstance(self.env.action_space, gym.spaces.Discrete):
        #     raise TypeError("Action space must be Discrete.")

        self.model = model
        # self.trainer = pl.Trainer(gpus=AVAIL_GPUS)

    def __call__(self, tasks, ch_avail):
        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)

        done = False
        while not done:
            with torch.no_grad():  # TODO: unneeded in PL?
                input_ = torch.from_numpy(obs[np.newaxis]).float()  # TODO: tensor conversion in model?
                prob = self.model(input_).squeeze(0)

            try:
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)
            except ValueError:
                prob = self.env.mask_probability(prob)
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def summary(self, file=None):
        print('Env: ', end='', file=file)
        self.env.summary(file)
        print('Model\n---\n```', file=file)

        print(self.model, file=file)
        print('```', end='\n\n', file=file)

    def learn(self, n_batch_train, batch_size_train=1, n_batch_val=0, batch_size_val=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        # FIXME: make PL mimic device handling as in Torch class above!?

        if verbose >= 1:
            print("Generating training data...")
        x_train, y_train, *__ = self.env.data_gen_numpy(n_batch_train * batch_size_train, weight_func=weight_func,
                                                        verbose=verbose)
        # d_train = self.env.data_gen_numpy(n_batch_train * batch_size_train, weight_func=weight_func, verbose=verbose)
        # x_train, y_train = d_train[:2]

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val = self.env.data_gen_numpy(n_batch_val * batch_size_val, weight_func=weight_func, verbose=verbose)

        # x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
        x_train, x_val = map(partial(torch.tensor, dtype=torch.float32), (x_train, x_val))
        y_train, y_val = map(partial(torch.tensor, dtype=torch.int64), (y_train, y_val))

        ds_train = TensorDataset(x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size_train * self.env.steps_per_episode, shuffle=True,
                              pin_memory=True)
        # FIXME: shuffle control? Enforce False??

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, batch_size=batch_size_val * self.env.steps_per_episode, shuffle=True,
                            pin_memory=True)

        if verbose >= 1:
            print('Training model...')

        trainer = pl.Trainer(gpus=AVAIL_GPUS, max_epochs=fit_params['epochs'])
        trainer.fit(self.model, dl_train, dl_val)
        # self.trainer.fit(self.model, dl_train, dl_val)

        # TODO: loss/acc plots, tensorboard, etc. LIGHTNING??

    def reset(self):
        self.model.apply(weights_init)

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
    #     return cls(model, env)  # FIXME: opt, loss, etc.? Easier with Lightning???
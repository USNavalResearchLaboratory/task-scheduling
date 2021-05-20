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


# from tensorboard import program

from task_scheduling.learning import environments as envs

# TODO: make loss func for full seq targets?
# TODO: make custom output layers to avoid illegal actions?


def weights_init(model):
    # if isinstance(m, nn.Conv2d):
    #     torch.nn.init.xavier_uniform(m.weight.data)
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class Scheduler:
    log_dir = Path.cwd() / 'logs' / 'torch_train'

    def __init__(self, model, env, loss_func, opt):
        self.model = model
        self.env = env

        self.loss_func = loss_func
        self.opt = opt

        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be Discrete.")

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

        ensure_valid = isinstance(self.env, envs.StepTasking) and not self.env.do_valid_actions
        # ensure_valid = False    # TODO: trained models may naturally avoid invalid actions!!

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)

        done = False
        with torch.no_grad():
            while not done:
                prob = self.model(torch.from_numpy(obs[np.newaxis])).numpy().squeeze(0)  # TODO: tensor conversion in model?
                # prob = self.model(obs[np.newaxis]).numpy().squeeze(0)  # FIXME
                # prob = self.model.predict_on_batch(obs[np.newaxis]).squeeze(0)
                if ensure_valid:
                    prob = self.env.mask_probability(prob)
                action = prob.argmax()

                obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def summary(self, file=None):
        print('Env: ', end='', file=file)
        self.env.summary(file)
        print('Model\n---\n```', file=file)
        print_fn = partial(print, file=file)
        # self.model.summary(print_fn=print_fn)  # FIXME
        print(self.model, file=file)
        print('```', end='\n\n', file=file)

    def learn(self, n_batch_train=1, n_batch_val=0, batch_size=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        if verbose >= 1:
            print("Generating training data...")
        d_train = self.env.data_gen_numpy(n_batch_train * batch_size, weight_func=weight_func, verbose=verbose)

        x_train, y_train = d_train[:2]
        x_train, y_train = map(torch.tensor, (x_train, y_train))

        ds_train = TensorDataset(x_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size * self.env.steps_per_episode, shuffle=True)
        # FIXME: shuffle control? Enforce False??

        # if callable(weight_func):  # FIXME: add sample weighting
        #     fit_params['sample_weight'] = d_train[2]

        if verbose >= 1:
            print("Generating validation data...")
        x_val, y_val = self.env.data_gen_numpy(n_batch_val * batch_size, weight_func=weight_func, verbose=verbose)
        x_val, y_val = map(torch.tensor, (x_val, y_val))

        ds_val = TensorDataset(x_val, y_val)
        dl_val = DataLoader(ds_val, shuffle=True)  # TODO: shuffle control

        # TODO: validation weighting?

        # # Add stopping callback if needed
        # if 'callbacks' not in fit_params:
        #     fit_params['callbacks'] = [keras.callbacks.EarlyStopping('val_loss', patience=60, min_delta=0.)]
        # elif not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in fit_params['callbacks']):
        #     fit_params['callbacks'].append(keras.callbacks.EarlyStopping('val_loss', patience=60, min_delta=0.))

        if verbose >= 1:
            print('Training model...')

        epochs = fit_params['epochs']
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in dl_train:
                loss = self.loss_func(self.model(xb), yb)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.loss_func(self.model(xb), yb) for xb, yb in dl_val)

            if verbose >= 2:
                print(f"  Epoch = {epoch} : loss = {valid_loss / len(dl_val):.3f}", end='\r')

        # TODO: train curves, tensorboard, etc.

    def reset(self):
        # reset_weights(self.model)  # FIXME
        self.model.apply(weights_init)

    def save(self, save_path=None):
        if save_path is None:
            save_path = f"models/temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        self.model.save(save_path)  # save TF model

        with Path(save_path).joinpath('env').open(mode='wb') as fid:
            dill.dump(self.env, fid)  # save environment

    @classmethod
    def load(cls, load_path):
        model = keras.models.load_model(load_path)

        with Path(load_path).joinpath('env').open(mode='rb') as fid:
            env = dill.load(fid)

        return cls(model, env)

    @classmethod
    def train_from_gen(cls, problem_gen, env_cls=envs.StepTasking, env_params=None, layers=None, compile_params=None,
                       n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None, fit_params=None,
                       do_tensorboard=False, plot_history=False, save=False, save_path=None):
        """
        Create and train a supervised learning scheduler.

        Parameters
        ----------
        problem_gen : generators.scheduling_problems.Base
            Scheduling problem generation object.
        env_cls : class, optional
            Gym environment class.
        env_params : dict, optional
            Parameters for environment initialization.
        layers : Sequence of tensorflow.keras.layers.Layer
            Neural network layers.
        compile_params : dict, optional
            Parameters for the model compile method.
        n_batch_train : int
            Number of batches of state-action pair data to generate for model training.
        n_batch_val : int
            Number of batches of state-action pair data to generate for model validation.
        batch_size : int
            Number of scheduling problems to make data from per yielded batch.
        weight_func : callable, optional
            Function mapping environment object to a training weight.
        fit_params : dict, optional
            Parameters for the mode fit method.
        do_tensorboard : bool, optional
            If True, Tensorboard is used for training visualization.
        plot_history : bool, optional
            If True, training is visualized using plotting modules.
        save : bool, optional
            If True, the network and environment are serialized.
        save_path : str, optional
            String representation of sub-directory to save to.

        Returns
        -------
        Scheduler

        """

        # Create environment
        if env_params is None:
            env = env_cls(problem_gen)
        else:
            env = env_cls(problem_gen, **env_params)

        # Create model
        if layers is None:
            layers = []

        model = keras.Sequential()
        model.add(keras.Input(shape=env.observation_space.shape))
        for layer in layers:  # add user-defined layers
            model.add(layer)
        if len(model.output_shape) > 2:  # flatten to 1-D for softmax output layer
            model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(env.action_space.n, activation='softmax',
                                     kernel_initializer=keras.initializers.GlorotUniform()))

        if compile_params is None:
            compile_params = {'optimizer': 'rmsprop',
                              'loss': 'sparse_categorical_crossentropy',
                              'metrics': ['accuracy'],
                              }
        model.compile(**compile_params)

        # Create and train scheduler
        scheduler = cls(model, env)
        scheduler.learn(n_batch_train, n_batch_val, batch_size, weight_func, fit_params, do_tensorboard, plot_history)
        if save:
            scheduler.save(save_path)

        return scheduler

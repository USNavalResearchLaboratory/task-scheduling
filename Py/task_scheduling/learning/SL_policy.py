import shutil
import time
import webbrowser
from functools import partial
from pathlib import Path

import dill
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard import program
from tensorflow import keras

from task_scheduling.learning import environments as envs


# TODO: make loss func for full seq targets?
# TODO: make custom output layers to avoid illegal actions?


class SupervisedLearningScheduler:
    log_dir = Path.cwd() / 'logs' / 'TF_train'

    def __init__(self, model, env):
        self.model = model
        self.env = env

        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be Discrete.")

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Iterable of task_scheduling.tasks.Base
        ch_avail : Iterable of float
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
        while not done:
            # if tf.executing_eagerly():
            #     prob = self.model(obs[np.newaxis]).numpy().squeeze(0)
            # else:
            #     a = 1 # TODO fix this problem. Actually run in eager execution, but getting other errors.
            #     # a_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
            #     # print(a_tensor)
            #     # an_array = a_tensor.eval(session=tf.compat.v1.Session())
            #     #
            #     # prob = self.model(obs[np.newaxis])
            #     # print(prob)
            #     # abc = prob.eval(session=tf.compat.v1.Session())
            #     #
            #     # # prob.eval(session=tf.compat.v1.Session())
            #     # sess = tf.Session()
            #     # with sess.as_default():
            #     #     # A = tf.constant([1, 2, 3]).eval()
            #     #     prob.eval(sess)

            prob = self.model(obs[np.newaxis]).numpy().squeeze(0)
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
        self.model.summary(print_fn=print_fn)
        print('```', end='\n\n', file=file)

    def fit(self, x, y=None, do_tensorboard=False, plot_history=False, **fit_params):

        if do_tensorboard:
            try:
                shutil.rmtree(self.log_dir)
            except FileNotFoundError:
                pass

            # fit_params['callbacks'].append(keras.callbacks.TensorBoard(log_dir=log_dir))
            fit_params['callbacks'] += [keras.callbacks.TensorBoard(log_dir=self.log_dir)]

            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', self.log_dir])
            url = tb.launch()
            webbrowser.open(url)

        # history = model.fit(d_train, **fit_params)      # generator Dataset
        history = self.model.fit(x, y, **fit_params)  # NumPy data

        acc_str = 'acc' if tf.version.VERSION[0] == '1' else 'accuracy'
        if plot_history:
            epoch = history.epoch
            if 'validation_freq' in fit_params:
                val_freq = fit_params['validation_freq']
                val_epoch = epoch[val_freq - 1:: val_freq]
            else:
                val_epoch = epoch
            hist_dict = history.history

            plt.figure(num='Training history', clear=True, figsize=(10, 4.8))
            plt.subplot(1, 2, 1)
            plt.plot(epoch, hist_dict['loss'], label='training')
            plt.plot(val_epoch, hist_dict['val_loss'], label='validation')

            plt.legend()
            plt.gca().set(xlabel='epoch', ylabel='loss')
            plt.subplot(1, 2, 2)
            plt.plot(epoch, hist_dict[acc_str], label='training')
            plt.plot(val_epoch, hist_dict['val_' + acc_str], label='validation')

            plt.legend()
            plt.gca().set(xlabel='epoch', ylabel='accuracy')

        return history

    def learn(self, n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None,
              fit_params=None, verbose=0, do_tensorboard=False, plot_history=False):

        d_val = self.env.data_gen_numpy(n_batch_val * batch_size, weight_func=weight_func, verbose=verbose)
        d_train = self.env.data_gen_numpy(n_batch_train * batch_size, weight_func=weight_func, verbose=verbose)

        x_train, y_train = d_train[:2]

        if callable(weight_func):
            sample_weight = d_train[2]
        else:
            sample_weight = None

        # gen_callable = partial(env.data_gen, weight_func=weight_func)  # function type not supported by from_generator
        #
        # output_types = (tf.float32, tf.int32)
        # output_shapes = ((None,) + env.observation_space.shape, (None,) + env.action_space.shape)
        # if callable(weight_func):
        #     output_types += (tf.float32,)
        #     output_shapes += ((None,),)
        #
        # d_train = tf.data.Dataset.from_generator(gen_callable, output_types,
        #                                          output_shapes, args=(n_batch_train, batch_size))
        # d_val = tf.data.Dataset.from_generator(gen_callable, output_types,
        #                                        output_shapes, args=(n_batch_val, batch_size))

        if fit_params is None:
            fit_params = {}
        fit_params.update({'validation_data': d_val,
                           # 'validation_freq': 1,
                           # 'batch_size': None,   # generator Dataset
                           'batch_size': batch_size * self.env.steps_per_episode,
                           'shuffle': False,
                           'sample_weight': sample_weight,
                           'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)],
                           'verbose': verbose - 1,
                           })

        if verbose >= 1:
            print('Training model...')

        # self.fit(*d_train, do_tensorboard, plot_history, **fit_params)
        self.fit(x_train, y_train, do_tensorboard, plot_history, **fit_params)

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
        layers : Iterable of tensorflow.keras.layers.Layer
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
        SupervisedLearningScheduler

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

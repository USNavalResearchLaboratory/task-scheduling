import shutil
from pathlib import Path
import time
import dill

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow import keras
from tensorboard import program
import webbrowser
import gym

from task_scheduling.learning import environments as envs

np.set_printoptions(precision=2)
plt.style.use('seaborn')

pkg_path = Path.cwd()
log_path = pkg_path / 'logs'
model_path = pkg_path / 'models'

# TODO: make loss func for full seq targets?
# TODO: make custom output layers to avoid illegal actions?


class SupervisedLearningScheduler:
    log_dir = log_path / 'TF_train'

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
        tasks : Iterable of tasks.Generic
        ch_avail : Iterable of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        # ensure_valid = isinstance(self.env, envs.StepTasking)
        ensure_valid = False    # TODO: trained models may naturally avoid invalid actions!!

        obs = self.env.reset(tasks, ch_avail)
        done = False
        while not done:
            prob = self.model.predict(obs[np.newaxis]).squeeze(0)

            if ensure_valid:
                seq_rem_sort = self.env.sorted_index_inv[list(self.env.node.seq_rem)]
                mask = np.isin(np.arange(self.env.n_tasks), seq_rem_sort, invert=True)

                prob = np.ma.masked_array(prob, mask)

            action = prob.argmax()
            obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

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

        if plot_history:
            plt.figure(num='training history', clear=True, figsize=(10, 4.8))
            plt.subplot(1, 2, 1)
            plt.plot(history.epoch, history.history['loss'], label='training')
            plt.plot(history.epoch, history.history['val_loss'], label='validation')
            plt.legend()
            plt.gca().set(xlabel='epoch', ylabel='loss')
            plt.subplot(1, 2, 2)
            plt.plot(history.epoch, history.history['accuracy'], label='training')
            plt.plot(history.epoch, history.history['val_accuracy'], label='validation')
            plt.legend()
            plt.gca().set(xlabel='epoch', ylabel='accuracy')

        return history

    def learn(self, n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None,
              fit_params=None, do_tensorboard=False, plot_history=False):

        # TODO: combine learn and fit methods?

        d_val = self.env.data_gen_numpy(n_batch_val * batch_size, weight_func=weight_func, verbose=True)
        d_train = self.env.data_gen_numpy(n_batch_train * batch_size, weight_func=weight_func, verbose=True)

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
                           # 'batch_size': None,   # generator Dataset
                           'batch_size': batch_size * self.env.steps_per_episode,
                           'shuffle': False,
                           'sample_weight': sample_weight,
                           'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]
                           })

        self.fit(x_train, y_train, do_tensorboard, plot_history, **fit_params)
        # self.fit(*d_train, do_tensorboard, plot_history, **fit_params)

    def save(self, save_path):
        if save_path is None:
            save_path = f"temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        save_path = model_path / save_path
        self.model.save(save_path)  # save TF model

        with save_path.joinpath('env').open(mode='wb') as fid:
            dill.dump(self.env, fid)  # save environment

    @classmethod
    def load(cls, load_path):
        load_path = model_path / load_path
        model = keras.models.load_model(load_path)

        with load_path.joinpath('env').open(mode='rb') as fid:
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
        env = env_cls.from_problem_gen(problem_gen, env_params)

        # Create model
        if layers is None:
            layers = [keras.layers.Dense(60, activation='relu'),
                      keras.layers.Dense(60, activation='relu'),
                      # keras.layers.Dense(30, activation='relu'),
                      # keras.layers.Dropout(0.2),
                      # keras.layers.Dense(100, activation='relu'),
                      ]

        model = keras.Sequential([keras.layers.Flatten(input_shape=env.observation_space.shape),
                                  *layers,
                                  keras.layers.Dense(env.action_space.n, activation='softmax')])

        if compile_params is None:
            compile_params = {'optimizer': 'rmsprop',
                              'loss': 'sparse_categorical_crossentropy',
                              'metrics': ['accuracy']
                              }
        model.compile(**compile_params)

        scheduler = cls(model, env)
        scheduler.learn(n_batch_train, n_batch_val, batch_size, weight_func, fit_params, do_tensorboard, plot_history)
        if save:
            scheduler.save(save_path)

        return scheduler

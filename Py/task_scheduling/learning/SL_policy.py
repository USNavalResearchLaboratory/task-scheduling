import shutil
import time
import dill

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorboard import program
import webbrowser
import gym

from generators.scheduling_problems import Dataset as ProblemDataset
from tree_search import TreeNodeShift
from learning.environments import SeqTaskingEnv, StepTaskingEnv

np.set_printoptions(precision=2)
plt.style.use('seaborn')


class SupervisedLearningScheduler:
    log_dir = '../../logs/TF_train'

    def __init__(self, model, env):
        self.model = model

        if isinstance(env, SeqTaskingEnv) and env.action_type == 'seq':
            # class FullSeq(keras.losses.Loss):
            #     def __init__(self, name="full_seq"):
            #         super().__init__(name=name)
            #
            #     def call(self, y_true, y_pred):
            #         return None
            raise NotImplementedError  # TODO: make loss func for full seq targets?
        else:
            self.env = env

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

        obs = self.env.reset(tasks, ch_avail)
        done = False
        while not done:
            prob = self.model.predict(obs[np.newaxis]).squeeze(0)

            if isinstance(self.env, StepTaskingEnv):
                # seq_rem = self.env.infer_action_space(obs).elements.tolist()
                seq_rem = self.env.action_space.elements.tolist()

                # TODO: masked array?
                action = seq_rem[prob[seq_rem].argmax()]  # FIXME: make custom output layers to avoid illegal actions?
            else:
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

    def save(self, save_path):
        if save_path is None:
            save_path = f"temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        self.model.save('../models/' + save_path)  # save TF model

        with open('../models/' + save_path + '/env', 'wb') as file:
            dill.dump(self.env, file)  # save environment

    @classmethod
    def load(cls, load_path):
        model = keras.models.load_model('../models/' + load_path)

        with open('../models/' + load_path + '/env', 'rb') as file:
            env = dill.load(file)

        return cls(model, env)

    @classmethod
    def train_from_gen(cls, problem_gen, env_cls=StepTaskingEnv, env_params=None, layers=None, n_batch_train=1,
                       n_batch_val=1, batch_size=1, weight_func=None, fit_params=None, do_tensorboard=False,
                       plot_history=False, save=False, save_path=None):

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

        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n
        else:
            n_actions = len(env.action_space)

        model = keras.Sequential([keras.layers.Flatten(input_shape=env.observation_space.shape),
                                  *layers,
                                  keras.layers.Dense(n_actions, activation='softmax')])

        compile_params = {'optimizer': 'rmsprop',
                          'loss': 'sparse_categorical_crossentropy',
                          'metrics': ['accuracy']
                          }
        model.compile(**compile_params)     # TODO

        scheduler = cls(model, env)

        # Generate data and train
        d_val = env.data_gen_numpy(n_batch_val * batch_size, weight_func=weight_func, verbose=True)
        d_train = env.data_gen_numpy(n_batch_train * batch_size, weight_func=weight_func, verbose=True)

        x_train, y_train = d_train[0], d_train[1]
        if len(d_train) == 3:
            sample_weight = d_train[2]
        else:
            sample_weight = None

        # gen_callable = partial(env.data_gen, weight_func=weight_func)  # function type not supported for from_generator
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
                           'batch_size': batch_size * env.steps_per_episode,
                           'sample_weight': sample_weight,
                           'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]
                           })

        scheduler.fit(x_train, y_train, do_tensorboard, plot_history, **fit_params)
        # scheduler.fit(*d_train, do_tensorboard, plot_history, **fit_params)

        # Save
        if save:
            scheduler.save(save_path)

        return cls(model, env)


# TODO: delete?
# def train_policy(problem_gen, n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None,
#                  env_cls=StepTaskingEnv, env_params=None,
#                  model=None, compile_params=None, fit_params=None,
#                  do_tensorboard=False, plot_history=False, save=False, save_dir=None):
#     """
#     Train a policy network via supervised learning.
#
#     Parameters
#     ----------
#     problem_gen : generators.scheduling_problems.Base
#         Scheduling problem generation object.
#     n_batch_train : int
#         Number of batches of state-action pair data to generate for model training.
#     n_batch_val : int
#         Number of batches of state-action pair data to generate for model validation.
#     batch_size : int
#         Number of scheduling problems to make data from per yielded batch.
#     weight_func : callable, optional
#         Function mapping environment object to a training weight.
#     env_cls : class, optional
#         Gym environment class.
#     env_params : dict, optional
#         Parameters for environment initialization.
#     model : tf.keras.Model, optional
#         Neural network model.
#     compile_params : dict, optional
#         Parameters for the model compile method.
#     fit_params : dict, optional
#         Parameters for the mode fit method.
#     do_tensorboard : bool
#         If True, Tensorboard is used for training visualization.
#     plot_history : bool
#         If True, training is visualized using plotting modules.
#     save : bool
#         If True, the network and environment are serialized.
#     save_dir : str, optional
#         String representation of sub-directory to save to.
#
#     Returns
#     -------
#     function
#         Wrapped policy. Takes tasks and channel availabilities and produces task execution times/channels.
#
#     """
#
#     # Create environment
#     env = env_cls.from_problem_gen(problem_gen, env_params)
#
#     if isinstance(env, SeqTaskingEnv) and env.action_type == 'seq':
#         # class FullSeq(keras.losses.Loss):
#         #     def __init__(self, name="full_seq"):
#         #         super().__init__(name=name)
#         #
#         #     def call(self, y_true, y_pred):
#         #         return None
#         raise NotImplementedError       # TODO: make loss func for full seq targets?
#
#     # Instantiate and compile policy model
#     if model is None:
#         if isinstance(env.action_space, gym.spaces.Discrete):
#             n_actions = env.action_space.n
#         else:
#             n_actions = len(env.action_space)
#
#         model = keras.Sequential([keras.layers.Flatten(input_shape=env.observation_space.shape),
#                                   keras.layers.Dense(60, activation='relu'),
#                                   keras.layers.Dense(60, activation='relu'),
#                                   # keras.layers.Dense(30, activation='relu'),
#                                   # keras.layers.Dropout(0.2),
#                                   # keras.layers.Dense(100, activation='relu'),
#                                   keras.layers.Dense(n_actions, activation='softmax')])
#
#     if compile_params is None:
#         compile_params = {'optimizer': 'rmsprop',
#                           'loss': 'sparse_categorical_crossentropy',
#                           'metrics': ['accuracy']
#                           }
#
#     model.compile(**compile_params)
#
#     # Generate state-action data, train model
#
#     d_val = env.data_gen_numpy(n_batch_val * batch_size, weight_func=weight_func, verbose=True)
#     d_train = env.data_gen_numpy(n_batch_train * batch_size, weight_func=weight_func, verbose=True)
#
#     # gen_callable = partial(env.data_gen, weight_func=weight_func)   # function type not supported for from_generator
#     #
#     # output_types = (tf.float32, tf.int32)
#     # output_shapes = ((None,) + env.observation_space.shape, (None,) + env.action_space.shape)
#     # if callable(weight_func):
#     #     output_types += (tf.float32,)
#     #     output_shapes += ((None,),)
#     #
#     # d_train = tf.data.Dataset.from_generator(gen_callable, output_types,
#     #                                          output_shapes, args=(n_batch_train, batch_size))
#     # d_val = tf.data.Dataset.from_generator(gen_callable, output_types,
#     #                                        output_shapes, args=(n_batch_val, batch_size))
#
#     # TODO: save dataset to save on Env computation time?
#
#     if fit_params is None:
#         fit_params = {'epochs': 10,
#                       'validation_data': d_val,
#                       # 'batch_size': None,   # generator Dataset
#                       'batch_size': batch_size * env.steps_per_episode,
#                       'sample_weight': None,
#                       'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]
#                       }
#
#     if do_tensorboard:
#         log_dir = '../../logs/TF_train'
#         try:
#             shutil.rmtree(log_dir)
#         except FileNotFoundError:
#             pass
#
#         # fit_params['callbacks'].append(keras.callbacks.TensorBoard(log_dir=log_dir))
#         fit_params['callbacks'] += [keras.callbacks.TensorBoard(log_dir=log_dir)]
#
#         tb = program.TensorBoard()
#         tb.configure(argv=[None, '--logdir', log_dir])
#         url = tb.launch()
#         webbrowser.open(url)
#
#     # history = model.fit(d_train, **fit_params)      # generator Dataset
#     history = model.fit(*d_train, **fit_params)   # NumPy data
#
#     if plot_history:
#         plt.figure(num='training history', clear=True, figsize=(10, 4.8))
#         plt.subplot(1, 2, 1)
#         plt.plot(history.epoch, history.history['loss'], label='training')
#         plt.plot(history.epoch, history.history['val_loss'], label='validation')
#         plt.legend()
#         plt.gca().set(xlabel='epoch', ylabel='loss')
#         plt.subplot(1, 2, 2)
#         plt.plot(history.epoch, history.history['accuracy'], label='training')
#         plt.plot(history.epoch, history.history['val_accuracy'], label='validation')
#         plt.legend()
#         plt.gca().set(xlabel='epoch', ylabel='accuracy')
#
#     if save:
#         if save_dir is None:
#             save_dir = f"temp/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
#
#         model.save('../models/' + save_dir)      # save TF model
#         with open('../models/' + save_dir + '/env', 'wb') as file:
#             dill.dump(env, file)    # save environment
#
#     return wrap_policy(env, model)
#
#
# def load_policy(load_dir):
#     """Loads network model and environment, returns wrapped scheduling function."""
#     with open('../models/' + load_dir + '/env', 'rb') as file:
#         env = dill.load(file)
#     model = keras.models.load_model('../models/' + load_dir)
#
#     return wrap_policy(env, model)
#
#
# def wrap_policy(env, model):
#     """Generate scheduling function by running a policy on a single environment episode."""
#
#     if isinstance(model, str):
#         model = keras.models.load_model(model)
#
#     def scheduling_model(tasks, ch_avail):
#         obs = env.reset(tasks, ch_avail)
#         done = False
#         while not done:
#             prob = model.predict(obs[np.newaxis]).squeeze(0)
#
#             if isinstance(env, StepTaskingEnv):
#                 seq_rem = env.infer_action_space(obs).elements.tolist()
#                 action = seq_rem[prob[seq_rem].argmax()]    # FIXME: make custom output layers to avoid illegal actions?
#             else:
#                 action = prob.argmax()
#
#             obs, reward, done, info = env.step(action)
#
#         return env.node.t_ex, env.node.ch_ex
#
#     return scheduling_model


def main():
    # problem_gen = RandomProblem.relu_drop(n_tasks=8, n_ch=2)
    problem_gen = ProblemDataset.load('relu_c1t8_1000', iter_mode='once', shuffle_mode='once', rng=None)

    features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                         ('release time', lambda task: task.t_release,
                          (0., problem_gen.task_gen.param_lims['t_release'][1])),
                         ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                         ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                         ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                         ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.node.tasks[n].t_release

    # env_cls = SeqTaskingEnv
    env_cls = StepTaskingEnv

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': True,
                  # 'action_type': 'int',
                  'seq_encoding': 'binary',
                  }

    weight_func_ = None
    # def weight_func_(env):
    #     return (env.n_tasks - len(env.node.seq)) / env.n_tasks

    scheduler = train_policy(problem_gen, n_batch_train=990, n_batch_val=10, batch_size=1, weight_func=weight_func_,
                             env_cls=env_cls, env_params=env_params,
                             model=None, compile_params=None, fit_params=None,
                             do_tensorboard=False, plot_history=True, save=True, save_dir=None)

    (tasks, ch_avail), = problem_gen(n_gen=1)
    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()

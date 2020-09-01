import shutil
import time
import dill
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorboard import program
import webbrowser

from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNodeShift
from env_tasking import StepTaskingEnv

np.set_printoptions(precision=2)
plt.style.use('seaborn')


def train_policy(problem_gen, n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None,
                 env_cls=StepTaskingEnv, env_params=None,
                 model=None, compile_params=None, fit_params=None,
                 do_tensorboard=False, plot_history=False, save=False, save_dir=None):
    """
    Train a policy network via supervised learning.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    n_batch_train : int
        Number of batches of state-action pair data to generate for agent training.
    n_batch_val : int
        Number of batches of state-action pair data to generate for agent validation.
    batch_size : int
        Number of scheduling problems to make data from per yielded batch.
    weight_func : callable, optional
        Function mapping partial sequence length and number of tasks to a training weight.
    env_cls : BaseTaskingEnv or callable
        Gym environment class.
    env_params : dict, optional
        Parameters for environment initialization.
    model : tf.keras.Model, optional
        Neural network model.
    compile_params : dict, optional
        Parameters for the model compile method.
    fit_params : dict, optional
        Parameters for the mode fit method.
    do_tensorboard : bool
        If True, Tensorboard is used for training visualization.
    plot_history : bool
        If True, training is visualized using plotting modules.
    save : bool
        If True, the network and environment are serialized.
    save_dir : str, optional
        String representation of sub-directory to save to.

    Returns
    -------
    function
        Wrapped policy. Takes tasks and channel availabilities and produces task execution times/channels.

    """

    # TODO: make custom output layers to avoid illegal actions

    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(problem_gen, **env_params)

    # Generate state-action data pairs

    # gen_callable = env.data_gen
    gen_callable = partial(env.data_gen, weight_func=weight_func)       # function type not supported for from_generator

    output_types = (tf.float32, tf.int32)
    output_shapes = ((batch_size * env.n_tasks,) + env.observation_space.shape, (batch_size * env.n_tasks,))
    if callable(weight_func):
        output_types += (tf.float32,)
        output_shapes += ((batch_size * env.n_tasks,),)

    # output_types = (tf.float32, tf.int32)
    # output_shapes = ((env.n_tasks,) + env.observation_space.shape, (env.n_tasks,))
    # output_shapes = (tf.TensorShape(env.observation_space.shape), tf.TensorShape(None))
    # output_shapes = ((batch_size * env.n_tasks,) + env.observation_space.shape, (batch_size * env.n_tasks,))

    d_train = tf.data.Dataset.from_generator(gen_callable, output_types,
                                             output_shapes, args=(n_batch_train, batch_size))
    d_val = tf.data.Dataset.from_generator(gen_callable, output_types,
                                           output_shapes, args=(n_batch_val, batch_size))

    # TODO: save dataset?

    # Train policy model
    if model is None:
        model = keras.Sequential([keras.layers.Flatten(input_shape=env.observation_space.shape),
                                  keras.layers.Dense(60, activation='relu'),
                                  keras.layers.Dense(60, activation='relu'),
                                  # keras.layers.Dense(30, activation='relu'),
                                  # keras.layers.Dropout(0.2),
                                  # keras.layers.Dense(100, activation='relu'),
                                  keras.layers.Dense(env.n_tasks, activation='softmax')])

    if compile_params is None:
        compile_params = {'optimizer': 'rmsprop',
                          'loss': 'sparse_categorical_crossentropy',
                          'metrics': ['accuracy']
                          }

    if fit_params is None:
        fit_params = {'epochs': 20,
                      'validation_data': d_val,
                      'batch_size': None,   # generator Dataset
                      'sample_weight': None,
                      'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]
                      }

    model.compile(**compile_params)

    if do_tensorboard:
        log_dir = '../logs/TF_train'
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        # fit_params['callbacks'].append(keras.callbacks.TensorBoard(log_dir=log_dir))
        fit_params['callbacks'] += [keras.callbacks.TensorBoard(log_dir=log_dir)]

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        webbrowser.open(url)

    history = model.fit(d_train, **fit_params)

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

    if save:
        if save_dir is None:
            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        model.save('../models/' + save_dir)      # save TF model
        with open('../models/' + save_dir + '/env.pkl', 'wb') as file:
            dill.dump(env, file)    # save environment

    return wrap_policy(env, model)


def load_policy(load_dir):
    """Loads network model and environment, returns wrapped scheduling function."""
    with open('../models/' + load_dir + '/env.pkl', 'rb') as file:
        env = dill.load(file)
    model = keras.models.load_model('models/' + load_dir)

    return wrap_policy(env, model)


def wrap_policy(env, model):
    """Generate scheduling function by running a policy on a single environment episode."""

    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")

    if type(model) == str:
        model = keras.models.load_model(model)

    def scheduling_model(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            prob = model.predict(observation[np.newaxis]).squeeze(0)
            seq_rem = env.action_space.elements.tolist()
            action = seq_rem[prob[seq_rem].argmax()]        # FIXME: hacked to disallow previously scheduled tasks

            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_model


def main():
    problem_gen = RandomProblem.relu_drop_default(n_tasks=4, n_ch=2)

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

    env_cls = StepTaskingEnv
    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'seq_encoding': 'indicator',
                  'masking': True
                  }

    def weight_func_(i, n):
        return (n - i) / n

    scheduler = train_policy(problem_gen, n_batch_train=5, n_batch_val=2, batch_size=2, weight_func=weight_func_,
                             env_cls=env_cls, env_params=env_params,
                             model=None, compile_params=None, fit_params=None,
                             do_tensorboard=False, plot_history=True, save=False, save_dir=None)

    (tasks, ch_avail), = problem_gen(n_gen=1)

    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()


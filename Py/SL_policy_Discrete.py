import shutil
import time
from functools import partial
import dill

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorboard import program
import webbrowser
import os

from util.generic import check_rng, Distribution
from util.results import check_valid, eval_loss

from generators.tasks import ContinuousUniformIID as ContinuousUniformTaskGenerator
from generators.tasks import GenericIID as GenericTaskGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release, TreeNode, TreeNodeShift
from env_tasking import StepTaskingEnv

np.set_printoptions(precision=2)
plt.style.use('seaborn')


def train_policy(n_tasks, task_gen, n_ch, ch_avail_gen,
                 n_gen_train=1, n_gen_val=1, env_cls=StepTaskingEnv, env_params=None,
                 model=None, compile_params=None, fit_params=None,
                 do_tensorboard=False, plot_history=True, save=False, save_dir=None):
    """
    Train a policy network via supervised learning.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    task_gen : GenericTaskGenerator
        Task generation object.
    n_ch: int
        Number of channels.
    ch_avail_gen : callable
        Returns random initial channel availabilities.
    n_gen_train : int
        Number of tasking problems to generate for agent training.
    n_gen_val : int
        Number of tasking problems to generate for agent validation.
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
    scheduler : function
        Wrapped policy. Takes tasks and channel availabilities and produces task execution times/channels.

    """

    # TODO: don't pack TF params in func arguments? user has to define all or none...
    # TODO: customize output layers to avoid illegal actions

    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(n_tasks, task_gen, n_ch, ch_avail_gen, **env_params)

    # Generate state-action data pairs
    d_train = env.data_gen(n_gen_train)
    d_val = env.data_gen(n_gen_val)

    # Train policy model
    if model is None:
        model = keras.Sequential([keras.layers.Flatten(input_shape=env.observation_space.shape),
                                  keras.layers.Dense(60, activation='relu'),
                                  keras.layers.Dense(60, activation='relu'),
                                  keras.layers.Dense(30, activation='relu'),
                                  keras.layers.Dropout(0.2),
                                  # keras.layers.Dense(100, activation='relu'),
                                  keras.layers.Dense(n_tasks, activation='softmax')])
    elif model.upper() is 'CNN':
        # current_shape = np.append(x_train[0].shape, 1)
        input_shape = d_train[0].shape
        model = keras.Sequential([keras.layers.BatchNormalization(input_shape=input_shape),
                                  keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'),
                                  keras.layers.BatchNormalization(),
                                  # keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                                  keras.layers.Dropout(0.2),
                                  keras.layers.Conv2D(64, (2, 2), activation='relu'),
                                  keras.layers.BatchNormalization(),
                                  keras.layers.Conv2D(128, (2, 2), activation='relu'),
                                  keras.layers.BatchNormalization(),
                                  # keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                  keras.layers.Flatten(),
                                  keras.layers.Dense(128, activation='relu'),
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
        fit_params = {'epochs': 1000,
                      'batch_size': 32,
                      'sample_weight': None,
                      'validation_data': d_val,
                      'callbacks': [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]
                      }

    model.compile(**compile_params)

    if do_tensorboard:
        log_dir = './logs/TF_train'
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        fit_params['callbacks'].append(keras.callbacks.TensorBoard(log_dir=log_dir))

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        webbrowser.open(url)

    history = model.fit(*d_train, **fit_params)

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
            save_path = './models/' + save_dir
        if ~os.path.isdir(save_path):
            os.mkdir(save_path)

        model.save(save_path)      # save TF model

        with open('./models/' + save_dir + '/env.pkl', 'wb') as file:
            dill.dump(env, file)    # save environment

    return wrap_policy(env, model)


def load_policy(load_dir):
    """Loads network model and environment, returns wrapped scheduling function."""
    with open('./models/' + load_dir + '/env.pkl', 'rb') as file:
        env = dill.load(file)
    model = keras.models.load_model('./models/' + load_dir)

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
    n_tasks = 6
    n_channels = 1

    # distro = []
    # distro[0] = Distribution(feature_name='duration', type='discrete', values=np.array([18e-3, 36e-3]), probs=(.25, .75))
    # a = distro[0].gen_samps(size=10)
    # distro[1] = Distribution(feature_name='t_release', type='uniform', lims=(0, 6))
    # distro[2] = Distribution(feature_name='slope', type='uniform', lims=(0.1, 2))
    # distro[3] = Distribution(feature_name='t_drop', type='uniform', lims=(2, 6))
    # distro[4] = Distribution(feature_name='l_drop', type='uniform', lims=(100, 300))

    # discrete_flag = np.array([True, False, False, False, False]) # Indicates whether feature is discrete or not
    # task_gen = ReluDropGenerator(duration_lim=(18e-3, 36e-3), t_release_lim=(0, 6), slope_lim=(0.1, 2),
    #                              t_drop_lim=(0, 15), l_drop_lim=(100, 300), rng=None,
    #                              discrete_flag=discrete_flag)  # task set generator

    def param_gen(self):
        return {'duration': self.rng.choice(self.param_lims['duration'], p=[.5, .5]),
                't_release': self.rng.uniform(self.param_lims['t_release']),
                'slope': self.rng.uniform(self.param_lims['slope']),
                't_drop': self.rng.uniform(self.param_lims['t_drop']),
                'l_drop': self.rng.uniform(self.param_lims['l_drop'])}

    param_lims = {'duration': (18e-3, 36e-3),
                  't_release': (0, 6),
                  'slope': (0.1, 2),
                  't_drop': (0, 15),
                  'l_drop': (100, 300)}

    task_gen = GenericTaskGenerator.relu_drop(param_gen, param_lims, rng=None)

    # task_gen = PermuteTaskGenerator(task_gen(n_tasks))
    # task_gen = DeterministicTaskGenerator(task_gen(n_tasks))

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(6, 6, n_ch)

    features = np.array([('duration', lambda self: self.duration, task_gen.duration_lim),
                         ('release time', lambda self: self.t_release, (0., task_gen.t_release_lim[1])),
                         ('slope', lambda self: self.slope, task_gen.slope_lim),
                         ('drop time', lambda self: self.t_drop, (0., task_gen.t_drop_lim[1])),
                         ('drop loss', lambda self: self.l_drop, (0., task_gen.l_drop_lim[1])),
                         ('is available', lambda self: 1 if self.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda self: 1 if self.l_drop == 0. else 0, (0, 1)),
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
                  'seq_encoding': 'indicator'
                  }

    n_total = 1000
    n_gen_train = np.round(n_total*.8).astype(int)
    n_gen_val = n_total - n_gen_train

    scheduler = train_policy(n_tasks, task_gen, n_channels, ch_avail_gen,
                             n_gen_train=n_gen_train, n_gen_val=n_gen_val,
                             env_cls=env_cls, env_params=env_params,
                             do_tensorboard=False, save=True)

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()

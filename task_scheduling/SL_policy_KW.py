import shutil
import time
import os

import pandas as pd
import numpy as np
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorboard import program
import webbrowser

from task_scheduling.util.generic import check_rng

from scheduling_algorithms import stats2nnXYgen
from task_scheduling.generators.scheduling_problems import ReluDrop
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.algorithms.free import branch_bound, branch_bound_with_stats
from learning.environments import StepTaskingEnv

plt.style.use('seaborn')


def data_gen(env, n_gen=1, gen_method=True):

    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")

    # TODO: generate sample weights to prioritize earliest task selections??

    x_gen = []
    y_gen = []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        env.reset()

        if gen_method:
            t_ex, ch_ex = branch_bound(env.node.tasks, env.node.ch_avail, verbose=True)
            seq = np.argsort(t_ex)     # optimal sequence
            # check_schedule(tasks, t_ex, ch_ex)
            # l_ex = evaluate_schedule(tasks, t_ex)

            for n in seq:
                x_gen.append(env.state.copy())
                y_gen.append(n)
                env.step(n)

        else:
            t_ex, ch_ex, NodeStats = branch_bound_with_stats(env.node.tasks, env.node.ch_avail)
            # [Xnow, Ynow] = stats2nnXY(NodeStats, env.node.tasks)
            [x_temp, y_temp] = stats2nnXYgen(NodeStats, env.node.tasks, env)
            x_gen = x_gen + x_temp
            y_gen = y_gen + y_temp


    return np.array(x_gen), np.array(y_gen)


def train_sl(env, n_gen_train, n_gen_val, plot_history=True, do_tensorboard=False, save_model=False, gen_method=False):

    # TODO: customize output layers to avoid illegal actions
    # TODO: train using complete tree info, not just B&B solution?

    # TODO: sort tasks by release time, etc.?
    # TODO: task parameter shift for channel availability

    x_train, y_train = data_gen(env, n_gen_train, gen_method)
    x_val, y_val = data_gen(env, n_gen_val, gen_method)
    x_train = x_train.reshape(np.append(x_train.shape, 1))
    x_val = x_val.reshape(np.append(x_val.shape, 1))

    # x_train = x_train.reshape(60000, 28, 28, 1)
    # x_test = x_test.reshape(10000, 28, 28, 1)

    # current_shape = np.append(x_train[0].shape, 1)
    input_shape = x_train[0].shape
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



    # model = keras.Sequential([keras.layers.Flatten(input_shape=x_train.shape[1:]),
    #                           keras.layers.Dense(60, activation='relu'),
    #                           keras.layers.Dense(60, activation='relu'),
    #                           # keras.layers.Dense(30, activation='relu'),
    #                           # keras.layers.Dropout(0.2),
    #                           # keras.layers.Dense(100, activation='relu'),
    #                           keras.layers.Dense(env.n_tasks, activation='softmax')])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', min_delta=0.)]

    if do_tensorboard:
        log_dir = 'logs/learn/tf'
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        webbrowser.open(url)



    history = model.fit(x_train, y_train, epochs=1000, batch_size=100, sample_weight=None,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

    if plot_history:
        # plt.figure(num='training history', clear=True, figsize=(10, 4.8))
        plt.figure(figsize=(10, 4.8))

        plt.subplot(1, 2, 1)
        plt.plot(history.epoch, history.history['loss'], label='training')
        plt.plot(history.epoch, history.history['val_loss'], label='validation')
        plt.legend()
        plt.gca().set(xlabel='epoch', ylabel='loss')
        plt.ylim(0,2)
        plt.title('Training Samples: ' + str(len(x_train)) )
        plt.subplot(1, 2, 2)
        plt.plot(history.epoch, history.history['accuracy'], label='training')
        plt.plot(history.epoch, history.history['val_accuracy'], label='validation')
        plt.legend()
        plt.gca().set(xlabel='epoch', ylabel='accuracy')
        plt.ylim(0, 1)
        plt.show()

    if save_model:      # TODO: pickle model and env together in dict? or just wrapped model func??
        save_str = 'models/temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
        if os.path.isdir(save_str) == False:
            os.mkdir(save_str)
        model.save(save_str)

    return model


def wrap_model(env, model):
    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")

    if type(model) == str:
        model = keras.models.load_model(model)

    def scheduling_model(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            obs = observation.reshape(np.append(observation.shape, 1))
            obs = obs.reshape(np.append(1, obs.shape))
            p = model.predict(obs).squeeze(0)
            # p = model.predict(observation[np.newaxis]).squeeze(0)

            seq_rem = env.action_space.elements.tolist()
            action = seq_rem[p[seq_rem].argmax()]        # FIXME: hacked to disallow previously scheduled tasks

            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_model


def main():
    n_tasks = 4
    n_channels = 1

    task_gen = ReluDrop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                        t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

    # task_gen = Permutation(task_gen(n_tasks))
    # task_gen = Deterministic(task_gen(n_tasks))

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 0, n_ch)

    env = StepTaskingEnv(n_tasks, task_gen, n_channels, ch_avail_gen, node_cls=TreeNodeShift, seq_encoding='one-hot')
    # env = StepTasking(n_tasks, task_gen, n_ch, ch_avail_gen, node_cls=TreeNode, seq_encoding='one-hot')


    n_total = 1000
    n_gen_train = np.int(np.round(n_total*.8))
    n_gen_val = np.int(n_total - n_gen_train)


    model = train_sl(env, n_gen_train=n_gen_train, n_gen_val=n_gen_val, plot_history=True, do_tensorboard=False, save_model=True, gen_method=False)
    # model = train_sl(env, n_gen_train=n_gen_train, n_gen_val=n_gen_val, plot_history=True, do_tensorboard=False, save_model=True, gen_method=True)

    # model = 'models/2020-07-09_08-39-48'

    scheduler = wrap_model(env, model)

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()

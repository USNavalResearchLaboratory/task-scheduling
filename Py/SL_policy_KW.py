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

from util.generic import check_rng
from util.results import check_valid, eval_loss

from scheduling_algorithms import stats2nnXY
from tasks import ReluDropGenerator, PermuteTaskGenerator, DeterministicTaskGenerator
from tree_search import branch_bound, branch_bound_with_stats,mcts_orig, mcts, random_sequencer, earliest_release, TreeNode, TreeNodeShift
from env_tasking import StepTaskingEnv

plt.style.use('seaborn')


def data_gen(env, n_gen=1):

    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")

    # TODO: generate sample weights to prioritize earliest task selections??

    x_gen = []
    y_gen = []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        env.reset()

        t_ex, ch_ex = branch_bound(env.node.tasks, env.node.ch_avail, verbose=True)
        t_ex, ch_ex, NodeStats = branch_bound_with_stats(env.node.tasks, env.node.ch_avail)
        [Xnow, Ynow] = stats2nnXY(NodeStats, env.node.tasks)

        seq = np.argsort(t_ex)     # optimal sequence

        # check_valid(tasks, t_ex, ch_ex)
        # l_ex = eval_loss(tasks, t_ex)

        for n in seq:
            x_gen.append(env.state.copy())
            y_gen.append(n)

            env.step(n)

    return np.array(x_gen), np.array(y_gen)


def train_sl(env, n_gen_train, n_gen_val, plot_history=True, do_tensorboard=False, save_model=False):

    # TODO: customize output layers to avoid illegal actions
    # TODO: train using complete tree info, not just B&B solution?

    # TODO: sort tasks by release time, etc.?
    # TODO: task parameter shift for channel availability

    x_train, y_train = data_gen(env, n_gen_train)
    x_val, y_val = data_gen(env, n_gen_val)

    model = keras.Sequential([keras.layers.Flatten(input_shape=x_train.shape[1:]),
                              keras.layers.Dense(60, activation='relu'),
                              keras.layers.Dense(60, activation='relu'),
                              # keras.layers.Dense(30, activation='relu'),
                              # keras.layers.Dropout(0.2),
                              # keras.layers.Dense(100, activation='relu'),
                              keras.layers.Dense(env.n_tasks, activation='softmax')])

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=60, monitor='val_loss', min_delta=0.)]

    if do_tensorboard:
        log_dir = './logs/TF_train'
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        webbrowser.open(url)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=32, sample_weight=None,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

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

    if save_model:      # TODO: pickle model and env together in dict? or just wrapped model func??
        save_str = './models/temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
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
            p = model.predict(observation[np.newaxis]).squeeze(0)

            seq_rem = env.action_space.elements.tolist()
            action = seq_rem[p[seq_rem].argmax()]        # FIXME: hacked to disallow previously scheduled tasks

            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_model


def main():
    n_tasks = 4
    n_channels = 1

    task_gen = ReluDropGenerator(t_release_lim=(0, 4), duration_lim=(3, 6), slope_lim=(0.5, 2),
                                 t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

    # task_gen = PermuteTaskGenerator(task_gen(n_tasks))
    # task_gen = DeterministicTaskGenerator(task_gen(n_tasks))

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 0, n_ch)

    env = StepTaskingEnv(n_tasks, task_gen, n_channels, ch_avail_gen, cls_node=TreeNodeShift, state_type='one-hot')

    model = train_sl(env, n_gen_train=10, n_gen_val=1, plot_history=True, do_tensorboard=False, save_model=True)
    # model = './models/2020-07-09_08-39-48'

    scheduler = wrap_model(env, model)

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()

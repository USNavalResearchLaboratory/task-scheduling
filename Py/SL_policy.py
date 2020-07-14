import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorboard import program
import webbrowser

from util.generic import check_rng
from util.results import check_valid, eval_loss

from tasks import ReluDropGenerator, PermuteTaskGenerator, DeterministicTaskGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release, TreeNode

plt.style.use('seaborn')


def data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen=1):

    # TODO: generate sample weights to prioritize earliest task selections??

    x_gen = []
    y_gen = []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        tasks = task_gen(n_tasks)
        ch_avail = ch_avail_gen(n_channels)

        t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)
        seq = np.argsort(t_ex).tolist()  # optimal sequence

        # check_valid(tasks, t_ex, ch_ex)
        # l_ex = eval_loss(tasks, t_ex)

        state = np.array([[0 for _ in range(n_tasks)] + task.gen_rep for task in tasks])        # one-hot
        # state = np.array([[1] + task.gen_rep for task in tasks])      # binary
        # state = np.ones((n_tasks, 1))     # no task parameters

        x = np.empty((n_tasks, *state.shape))
        y = np.zeros(n_tasks, dtype=np.int)
        for i, n in enumerate(seq):
            x[i] = state
            y[i] = n

            state[n][i] = 1
            # state[n][0] = 0

        x_gen.append(x)
        y_gen.append(y)

    return np.concatenate(x_gen), np.concatenate(y_gen)


def train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train, n_gen_val,
             plot_history=True, do_tensorboard=False, save_model=False):

    # TODO: customize output layers to avoid illegal actions
    # TODO: train using complete tree info, not just B&B solution?

    # TODO: sort tasks by release time, etc.?
    # TODO: task parameter shift for channel availability

    x_train, y_train = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train)
    x_val, y_val = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_val)

    model = keras.Sequential([keras.layers.Flatten(input_shape=x_train.shape[1:]),
                              keras.layers.Dense(60, activation='relu'),
                              keras.layers.Dense(60, activation='relu'),
                              # keras.layers.Dense(30, activation='relu'),
                              # keras.layers.Dropout(0.2),
                              # keras.layers.Dense(100, activation='relu'),
                              keras.layers.Dense(n_tasks, activation='softmax')])

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

    if save_model:
        model.save('./models/temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))

    return model


def wrap_model(model):
    if type(model) == str:
        model = keras.models.load_model(model)

    def scheduling_model(tasks, ch_avail):
        TreeNode._tasks = tasks
        TreeNode._ch_avail_init = ch_avail

        state = np.array([[1] + task.gen_rep for task in tasks])
        seq = []
        for _ in range(len(tasks)):
            p = model.predict(state[np.newaxis]).squeeze(0)
            p[seq] = 0.     # FIXME: hacked to disallow previously scheduled tasks
            n = p.argmax()

            seq.append(n)
            state[n][0] = 0

        node = TreeNode(seq)
        return node.t_ex, node.ch_ex

    return scheduling_model


def main():
    n_tasks = 8
    n_channels = 2

    task_gen = ReluDropGenerator(t_release_lim=(0, 4), duration_lim=(3, 6), slope_lim=(0.5, 2),
                                 t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

    # task_gen = PermuteTaskGenerator(task_gen(n_tasks))
    # task_gen = DeterministicTaskGenerator(task_gen(n_tasks))

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        return rng.uniform(0, 0, n_ch)

    # x, y = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen=10)

    model = train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train=100, n_gen_val=10,
                     plot_history=True, do_tensorboard=False, save_model=True)
    # model = './models/2020-07-09_08-39-48'

    scheduler = wrap_model(model)

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels)

    t_ex, ch_ex = scheduler(tasks, ch_avail)

    print(t_ex)
    print(ch_ex)


if __name__ == '__main__':
    main()

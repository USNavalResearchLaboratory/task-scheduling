import shutil

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorboard import program
import webbrowser

from util.generic import check_rng
from util.results import check_valid, eval_loss

from tasks import ReluDropGenerator, PermuteTaskGenerator, DeterministicTaskGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release

plt.style.use('seaborn')


def data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen=1):

    # TODO: generate sample weights to prioritize earliest task selections??

    x_gen = []
    y_gen = []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        tasks = task_gen.rand_tasks(n_tasks)
        ch_avail = ch_avail_gen(n_channels)

        t_ex, ch_ex = branch_bound(tasks, ch_avail, verbose=True)

        # check_valid(tasks, t_ex, ch_ex)
        # l_ex = eval_loss(tasks, t_ex)

        seq = np.argsort(t_ex).tolist()     # optimal sequence
        state = np.array([[1] + task.gen_rep for task in tasks])

        x = np.empty((n_tasks, *state.shape))
        # x = np.empty((n_tasks, n_tasks))
        y = np.zeros(n_tasks, dtype=np.int)
        for i, n in enumerate(seq):
            x[i] = state
            # x[i] = state[:, 0]
            y[i] = n

            state[n][0] = 0

        x_gen.append(x)
        y_gen.append(y)

    x_full = np.concatenate(x_gen)
    y_full = np.concatenate(y_gen)

    return x_full, y_full


def train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train, n_gen_val, do_tensorboard=False):

    x_train, y_train = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train)
    x_val, y_val = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_val)

    model = keras.Sequential([keras.layers.Flatten(input_shape=x_train.shape[1:]),
                              keras.layers.Dense(60, activation='relu'),
                              # keras.layers.Dense(30, activation='relu'),
                              # keras.layers.Dense(30, activation='relu'),
                              # keras.layers.Dropout(0.2),
                              # keras.layers.Dense(100, activation='relu'),
                              keras.layers.Dense(n_tasks, activation='softmax')])

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=100, monitor='val_loss', min_delta=0.)]

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

    history = model.fit(x_train, y_train, epochs=2000, batch_size=32, sample_weight=None,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # model.save_weights('./weights/my_model')

    if True:
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

    return model    # TODO: wrap model for main.py


def main():
    n_tasks = 5
    n_channels = 2

    task_gen = ReluDropGenerator(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                                 t_drop_lim=(6, 12), l_drop_lim=(35, 50), rng=None)  # task set generator

    # task_gen = PermuteTaskGenerator(task_gen.rand_tasks(n_tasks))   # FIXME
    # task_gen = DeterministicTaskGenerator(task_gen.rand_tasks(n_tasks))  # FIXME

    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        # TODO: rng is a mutable default argument!
        # return rng.uniform(0, 2, n_ch)
        return np.zeros(n_ch)

    # x, y = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen=10)

    train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train=100, n_gen_val=10)


if __name__ == '__main__':
    main()

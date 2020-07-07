import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from util.generic import check_rng
from util.results import check_valid, eval_loss

from tasks import ReluDropGenerator, PermuteTaskGenerator, DeterministicTaskGenerator
from tree_search import branch_bound, mcts_orig, mcts, random_sequencer, earliest_release

plt.style.use('seaborn')


def obs_relu_drop(tasks):
    """Generate observation array from list of tasks."""

    # _params = [(task.duration, task.t_release, task.slope, task.t_drop, task.l_drop) for task in tasks]
    # params = np.array(_params, dtype=[('duration', np.float), ('t_release', np.float),
    #                                   ('slope', np.float), ('t_drop', np.float), ('l_drop', np.float)])
    # params.view(np.float).reshape(*params.shape, -1)
    return np.asarray([[task.duration, task.t_release, task.slope, task.t_drop, task.l_drop] for task in tasks])


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
        state = np.concatenate((np.ones((n_tasks, 1)), obs_relu_drop(tasks)), axis=1)

        x = np.empty((n_tasks, *state.shape))
        # x = np.empty((n_tasks, n_tasks))
        # y = np.zeros((n_tasks, n_tasks), dtype=np.int)
        y = np.zeros(n_tasks, dtype=np.int)
        for i, n in enumerate(seq):
            x[i] = state
            # x[i] = state[:, 0]
            # y[i][n] = 1
            y[i] = n

            state[n][0] = 0

        x_gen.append(x)
        y_gen.append(y)

    x_full = np.concatenate(x_gen)
    y_full = np.concatenate(y_gen)

    return x_full, y_full


def train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train, n_gen_val):
    x_train, y_train = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train)
    x_val, y_val = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_val)

    model = keras.Sequential([keras.layers.Flatten(input_shape=(n_tasks, 6)),
                              keras.layers.Dense(100, activation='relu'),
                              # keras.layers.Dropout(0.2),
                              # keras.layers.Dense(100, activation='relu'),
                              keras.layers.Dense(n_tasks, activation=None)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
                 # keras.callbacks.TensorBoard(log_dir='./logs')]

    history = model.fit(x_train, y_train, epochs=500, batch_size=32, sample_weight=None,
                        validation_data=(x_val, y_val),
                        callbacks=None)

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
    task_gen = DeterministicTaskGenerator(task_gen.rand_tasks(n_tasks))  # FIXME


    def ch_avail_gen(n_ch, rng=check_rng(None)):  # channel availability time generator
        # TODO: rng is a mutable default argument!
        # return rng.uniform(0, 2, n_ch)
        return np.zeros(n_ch)

    # x, y = data_gen(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen=10)

    train_sl(task_gen, n_tasks, ch_avail_gen, n_channels, n_gen_train=10, n_gen_val=1)


if __name__ == '__main__':
    main()

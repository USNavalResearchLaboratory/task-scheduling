"""
Task scheduling example.

Define a set of task objects and scheduling algorithms. Assess achieved loss and runtime.

"""

# TODO: Account for algorithm runtime before evaluating execution loss!!

# TODO: limit execution time of algorithms using signal module?
# TODO: add proper main() def, __name__ conditional execution?



import time     # TODO: use builtin module timeit instead? or cProfile?
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scheduling_algorithms import stats2nnXY

np.set_printoptions(linewidth=300) # Set printing to avoid line wrapping when displaying 2D array

from task_scheduling.util.generic import algorithm_repr, check_rng
from task_scheduling.util.results import check_valid, eval_loss
from task_scheduling.util.plot import plot_task_losses, scatter_loss_runtime

from task_scheduling.generators.scheduling_problems import ReluDrop
from task_scheduling.algorithms.base import branch_bound_with_stats, mcts_orig, earliest_release
from learning.environments import StepTaskingEnv, wrap_agent, RandomAgent

plt.style.use('seaborn')
rng = np.random.default_rng(100)
# rng = np.random.seed(100)

# %% Inputs
n_gen = 10      # number of task scheduling problems

n_tasks = 8
n_channels = 1

task_gen = ReluDrop(duration_lim=(3, 6), t_release_lim=(0, 4), slope_lim=(0.5, 2),
                    t_drop_lim=(12, 20), l_drop_lim=(35, 50), rng=rng)       # task set generator

def ch_avail_gen(n_ch, rng=check_rng(None)):     # channel availability time generator
    # TODO: rng is a mutable default argument!
    return rng.uniform(0, 2, n_ch)

# import cProfile
# with cProfile.Profile() as pr:
#     a = 2
# pr.print_stats()

# Algorithms

env = StepTaskingEnv(n_tasks, task_gen, n_channels, ch_avail_gen)
random_agent = wrap_agent(env, RandomAgent(env.infer_action_space))

alg_funcs = [partial(branch_bound_with_stats, verbose=False, rng = rng),
             # partial(branch_bound2, verbose=False, rng = rng),
             # partial(branch_bound_rules, verbose=False),
             partial(mcts_orig, n_mc=100, verbose=False),
             partial(earliest_release, do_swap=True)]#,
             # partial(random_sequencer)]#,
             # partial(random_agent)]

alg_n_runs = [1, 1, 1]       # number of runs per problem

alg_reprs = list(map(algorithm_repr, alg_funcs))


# %% Evaluate
t_run_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

l_ex_iter = np.array(list(zip(*[np.empty((n_gen, n_run)) for n_run in alg_n_runs])),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

# t_ex_iter = np.array(list(zip(*[np.empty((n_gen, n_run, n_tasks)) for n_run in alg_n_runs])),
#                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float], [(n_run,) for n_run in alg_n_runs])))

t_ex_alg = np.empty((n_gen, len(alg_reprs), np.max(alg_n_runs),n_tasks))
T_alg = np.empty((n_gen, len(alg_reprs), np.max(alg_n_runs),n_tasks))

t_run_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

t_run_mean2 = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                      dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

l_ex_mean2 = np.array(list(zip(*np.empty((len(alg_reprs), n_gen)))),
                     dtype=list(zip(alg_reprs, len(alg_reprs) * [np.float])))

X = np.empty([0,n_tasks+5+3,n_tasks])
Y = np.empty([0])

for i_gen in range(n_gen):      # Generate new tasks
    print(f'Task Set: {i_gen + 1}/{n_gen}')

    tasks = task_gen(n_tasks)
    ch_avail = ch_avail_gen(n_channels,rng)

    _, ax_gen = plt.subplots(2, 1, num=f'Task Set: {i_gen + 1}', clear=True)
    plot_task_losses(tasks, ax=ax_gen[0])

    for alg_repr, alg_func, n_run in zip(alg_reprs, alg_funcs, alg_n_runs):
        for i_run in range(n_run):      # Perform new algorithm runs
            print(f'  {alg_repr} - Run: {i_run + 1}/{n_run}', end='\r')

            t_start = time.time()
            if alg_repr == 'branch_bound_with_stats':
                t_ex, ch_ex, NodeStats = alg_func(tasks, ch_avail)
                [Xnow, Ynow] = stats2nnXY(NodeStats, tasks)
            else:
                t_ex, ch_ex = alg_func(tasks, ch_avail)

            X = np.concatenate((X, np.array(Xnow)), axis=0)
            Y = np.concatenate((Y, np.array(Ynow)), axis=0)
            # X.append(Xnow)
            # Y.append(Ynow)

            t_run = time.time() - t_start

            check_valid(tasks, t_ex, ch_ex)
            l_ex = eval_loss(tasks, t_ex)

            t_run_iter[alg_repr][i_gen, i_run] = t_run
            l_ex_iter[alg_repr][i_gen, i_run] = l_ex
            # t_ex_iter[name][i_gen, i_run,:] = t_ex
            t_ex_alg[i_gen, alg_reprs.index(alg_repr) , i_run, :] = t_ex
            T_alg[i_gen, alg_reprs.index(alg_repr), i_run, :] = np.argsort(t_ex)


            # plot_schedule(tasks, t_ex, ch_ex, l_ex=l_ex, name=name, ax=None)

        t_run_mean[alg_repr][i_gen] = t_run_iter[alg_repr][i_gen].mean()
        l_ex_mean[alg_repr][i_gen] = l_ex_iter[alg_repr][i_gen].mean()

        t_run_mean2[alg_repr] = t_run_iter[alg_repr].mean()
        l_ex_mean2[alg_repr] = l_ex_iter[alg_repr].mean()

        print('')
        print(f"    Avg. Runtime: {t_run_mean[alg_repr][i_gen]:.2f} (s)")
        print(f"    Avg. Execution Loss: {l_ex_mean[alg_repr][i_gen]:.2f}")

    scatter_loss_runtime(t_run_iter[i_gen], l_ex_iter[i_gen], ax=ax_gen[1])

print('')

_, ax_results = plt.subplots(num='Results', clear=True)
scatter_loss_runtime(t_run_mean, l_ex_mean, ax=ax_results, ax_kwargs={'title': 'Average performance on random task sets'})



_, ax_results2 = plt.subplots(num='Results', clear=True)
scatter_loss_runtime(t_run_mean2, l_ex_mean2, ax=ax_results2, ax_kwargs={'title': 'Average performance on random task sets'})

# Setup Training Data # TODO Need to make sure splits arent' across problems
Nsamp = len(X)
split = [0.75, 0.25]
Ntrain = np.rint(split[0]*Nsamp)
Ntrain = Ntrain.astype('int32')
Ntest = Nsamp - Ntrain
train_samples = np.sort(np.random.choice(np.arange(Nsamp),Ntrain,replace=False))
test_samples = np.setdiff1d(np.arange(Nsamp),train_samples)

Xtrain = np.expand_dims(X[train_samples,:,:],axis=3)
Ytrain = Y[train_samples]
Xtest = np.expand_dims(X[test_samples,:,:],axis=3)
Ytest = Y[test_samples]



# Train a Neural Network
import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # Very slow on a laptop --> lots of data downloaded

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(n_tasks+5+3, n_tasks,1)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(n_tasks))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))
history = model.fit(Xtrain, Ytrain, epochs=100,
                    validation_data=(Xtest, Ytest))


plt.figure(40)
plt.clf()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(Xtest,  Ytest, verbose=2)

print(test_acc)


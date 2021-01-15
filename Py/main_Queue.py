
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gym import spaces

import task_scheduling
from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms.base import earliest_release, branch_bound
from task_scheduling.learning import environments as envs
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.learning.RL_policy import ReinforcementLearningScheduler as RL_Scheduler
from task_scheduling.learning.features import _get_param
from task_scheduling.util.results import timing_wrapper, evaluate_algorithms, evaluate_algorithms_runtime, iter_to_mean
from task_scheduling.util.plot import scatter_loss_runtime_stats
from task_scheduling.tasks import check_task_types
from task_scheduling.learning.SL_policy import SupervisedLearningScheduler as SL_Scheduler
from tensorflow import keras
import tensorflow as tf
from task_scheduling.learning.features import param_features, encode_discrete_features
tf.enable_eager_execution()  # Set tf to eager execution --> avoids error in SL_policy line 61
# plt.style.use(['science', 'ieee']) # Used for plotting style ensures plots are visible in black and white
# plt.figure()
# plt.plot(np.arange(9))
# plt.xlabel('hello')
# plt.ylabel('B\&B')
# plt.savefig('abc.pdf')
# plt.show()

def generate_data(create_data_flag=False, n_gen=None, n_tasks=None, n_track=None, n_ch=None, setup_type=None):

    # create_data_flag = True

    if create_data_flag:

        # n_gen = 10000
        # n_tasks = 4  # Number of tasks to process at each iteration
        # n_track = 10
        # ch_avail = np.zeros(n_ch, dtype=np.float)
        if setup_type == 'FlexDARlike':
            tasks_full = task_gens.FlexDARlike(n_track=n_track, rng=100).tasks_full
        else:
            tasks_full = task_gens.FlexDAR(n_track=n_track, rng=100).tasks_full

        # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
        # ch_avail = [0, 0]
        ch_avail = [0]*n_ch
        problem_gen = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail, record_revisit=False)


        # features = np.array([('duration', lambda task: task.duration, (0, 10)),
        #                      ('release time', lambda task: task.t_release, (0, 10)),
        #                      ('slope', lambda task: task.slope, (0, 10)),
        #                      ('drop time', lambda task: task.t_drop, (0, 10)),
        #                      ('drop loss', lambda task: task.l_drop, (0, 10)),
        #                      ],
        #                     dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
        #
        # # env_cls = envs.SeqTasking
        # env_cls = envs.SeqTasking
        #
        # env_params = {'features': features,
        #               'sort_func': None,
        #               'time_shift': True,
        #               'masking': True,
        #               'action_type': 'int',
        #               # 'action_type': 'any',
        #               # 'seq_encoding': 'one-hot',
        #               }
        #
        # env = env_cls(problem_gen, **env_params)

        filename = setup_type + '_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track) + \
                   '_' + str(n_gen)
        list(problem_gen(n_gen=n_gen, save=True, file=filename))




plot_hist_flag = False
train_RL_flag = True
train_SL_flag = True
setup_type = 'FlexDARlike'  # Option 1: FlexDAR or FlexDARlike

n_gen = 100
n_train = np.array(n_gen*0.9, dtype=int)
n_eval = n_gen - n_train - 1
# n_train = 10000
# n_eval = 200

n_tasks = 5  # Number of tasks to process at each iteration
n_track = 10
# n_track_eval = 10
# ch_avail = np.zeros(2, dtype=np.float)
if setup_type == 'FlexDARlike':
    tasks_full = task_gens.FlexDARlike(n_track=n_track, rng=100).tasks_full
else:
    tasks_full = task_gens.FlexDAR(n_track=n_track, rng=100).tasks_full

# ch_avail = list(ch_gens.UniformIID((0, 0))(2))
n_ch = 1

ch_avail = [0 ] *n_ch
# n_ch = len(ch_avail)

# Problem Generator
# Use separate datasets for training and evaluation. Let Training dataset repeat for training.
filename_train = setup_type + '_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track) + '_' + str \
    (n_gen)
# Eval with 0 tracks for now
# filename_eval = 'FlexDAR_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track_eval) + '_' + str \
#     (n_eval)
filepath_train = './data/schedules/' + filename_train
# filepath_eval = './data/schedules/' + filename_eval
if os.path.isfile(filepath_train):
    problem_gen = problem_gens.Dataset.load(file=filename_train, shuffle=False, rng=None, repeat=True)
else:
    generate_data(create_data_flag=True, n_gen=n_gen, n_tasks=n_tasks, n_track=n_track, n_ch=n_ch, setup_type=setup_type)
    problem_gen = problem_gens.Dataset.load(file=filename_train, shuffle=False, rng=None, repeat=True)

# if os.path.isfile(filepath_eval):
#     problem_gen_eval = problem_gens.Dataset.load(file=filename_eval, shuffle=False, rng=None)
# else:
#     generate_data(create_data_flag=True, n_gen=n_eval, n_tasks=n_tasks, n_track=n_track_eval, n_ch=n_ch)
#     problem_gen_eval = problem_gens.Dataset.load(file=filename_eval, shuffle=False, rng=None)


n_problems = len(problem_gen.problems)

if plot_hist_flag:
    df = pd.DataFrame()
    for jj in range(n_problems):
        if jj % 100 == 0:
            print('Iteration ' + str(jj) + ' of ' + str(n_problems))

        tasks = problem_gen.problems[jj].tasks
        ch_avail = problem_gen.problems[jj].ch_avail
        cls_task = check_task_types(tasks)
        df2 = pd.DataFrame({name: [getattr(task, name) for task in tasks]
                            for name in cls_task.param_names})
        for kk in range(len(ch_avail)):
            name = 'ch' + str(kk)
            df2[name] = np.ones(shape=len(tasks)) * np.array(ch_avail[kk], dtype=float)
            name = 'ch_max'
            df2[name] = np.ones(shape=len(tasks)) * np.max(np.array(ch_avail, dtype=float))
            name = 'ch_min'
            df2[name] = np.ones(shape=len(tasks)) * np.min(np.array(ch_avail, dtype=float))

        df = df.append(df2)

    name = 't_release - ch_min'
    df[name] = df['t_release'] - df['ch_min']

    for col in df.columns:
        # plt.figure()
        # df.plot.hist(column=jj, bins=100, title=df.columns[jj], ax=jj)
        df.hist(column=col, bins=100)

# time_shift = False
# features = param_features(problem_gen, time_shift)


features = np.array([('duration', _get_param('duration'), spaces.Box(0, 0.05, shape=(), dtype=np.float)),
                     # ('t_release', get_param('t_release'),
                     #  (0., problem_gen.task_gen.param_lims['t_release'][1])),
                     ('slope', _get_param('slope'), spaces.Box(0, 1, shape=(), dtype=np.float)),
                     ('t_drop', _get_param('t_drop'), spaces.Box(0, 6, shape=(), dtype=np.float)),
                     ('offset', lambda tasks_, ch_avail_: [task.t_release - np.min(ch_avail_) for task in tasks_],
                      spaces.Box(-5, 0, shape=(), dtype=np.float)),
                     # ('l_drop', get_param('l_drop'), (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                     ],
                    dtype=[('name', '<U16'), ('func', object), ('space', object)])


# features = np.array([('duration', make_attr_feature('duration'), problem_gen.task_gen.param_lims['duration']),
#                      # ('t_release', make_attr_feature('t_release'),
#                      #  (0., problem_gen.task_gen.param_lims['t_release'][1])),
#                      ('slope', make_attr_feature('slope'), problem_gen.task_gen.param_lims['slope']),
#                      ('t_drop', make_attr_feature('t_drop'), (0., problem_gen.task_gen.param_lims['t_drop'][1])),
#                      ('offset', lambda tasks_, ch_avail_: [task.t_release - np.min(ch_avail_) for task in tasks_],
#                       (0., problem_gen.task_gen.param_lims['t_release'][1])),
#                      # ('l_drop', make_attr_feature('l_drop'), (0., problem_gen.task_gen.param_lims['l_drop'][1])),
#                      ],
#                     dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

# features = np.array([('duration', lambda task: task.duration, (0, 0.05)),
#                      # ('release time', lambda task: task.t_release, (0, 1)),
#                      # ('release time', lambda task: task.t_release - , (0, 10)),
#                      ('slope', lambda task: task.slope, (0, 1)),
#                      ('drop time', lambda task: task.t_drop, (0, 6)),
#                      ('offset', lambda task: task.t_release - np.min(task.ch_avail), (-5, 0)),
#
#                      # ('drop loss', lambda task: task.l_drop, (0, 10)),
#                      ],
#                     dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

# env_cls = envs.SeqTasking
env_cls = envs.SeqTasking

env_params = {'features': features,
              'sort_func': None,
              'time_shift': False,
              'masking': False,
              'action_type': 'int',
              # 'action_type': 'any',
              # 'seq_encoding': 'one-hot',
              }

env = env_cls(problem_gen, **env_params)
# env.reset()


# env.problem_gen(1, solve=False)
# env.reset()
# for __ in range(10):
#     (tasks, ch_avail), = env.problem_gen(1, solve=False)

# Train RL Approach
if train_RL_flag:
    env_cls_RL = envs.SeqTasking
    dqn_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls_RL, env_params,
                                            # model_cls='DQN_LN', model_params={'verbose': 1}, n_episodes=n_train * 10,
                                            # model_cls='CNN', model_params={'verbose': 1}, n_episodes=n_train*100,
                                            model_cls='DQN', model_params={'verbose': 1}, n_episodes=n_train * 1,
                                            save=False, save_path='./')


# Train SL Approach
if train_SL_flag:
    env_cls_SL = envs.SeqTasking

    # layers = None
    input_shape = (None, 20)
    layers = [
              # keras.layers.Conv1D(filters=4, kernel_size=3, padding='same', activation='relu',input_shape=input_shape[1:]),
              keras.layers.Dense(20, activation='relu'),
              keras.layers.Dense(10, activation='relu'),
              # keras.layers.Dense(30, activation='relu'),
              # keras.layers.Dropout(0.2),
              # keras.layers.Dense(100, activation='relu'),
              ]
    weight_func_ = None

    policy_model = SL_Scheduler.train_from_gen(problem_gen, env_cls_SL, env_params, layers=layers, compile_params=None,
                                               n_batch_train=35, n_batch_val=10, batch_size=20, weight_func=weight_func_,
                                               fit_params={'epochs': 300}, do_tensorboard=False, plot_history=True,
                                               save=False, save_path=None)



algorithms = np.array([
    # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
    # ('Random', algs_base.random_sequencer, 20),
    ('ERT', earliest_release, 1),
    # ('MCTS', partial(algs_base.mcts, n_mc=100, verbose=False), 5),
    # if train_RL_flag:
        ('DQN Agent', dqn_agent, 1),
    # if train_SL_flag:
        ('DNN Policy', policy_model, 1),
], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

# l_ex_iter, t_run_iter, l_ex_mean, t_run_mean, l_ex_mean_norm = evaluate_algorithms(algorithms, problem_gen,
#                                                                                    n_gen=n_eval, solve=True,
#                                                                                    verbose=2, plotting=1,
#                                                                                    save=False, file=None)

l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, problem_gen, n_gen=n_eval, solve=True,
                                            verbose=2, plotting=1, save=False, file=None)

l_ex_mean, t_run_mean = map(iter_to_mean, (l_ex_iter, t_run_iter))
scatter_loss_runtime_stats(t_run_mean, l_ex_mean, ax=None, ax_kwargs=None)



##############################################################
# Evaluate Scheduler in context of Radar Queue
solve=True
if solve:
    _opt = np.array([('BB Optimal', branch_bound, 1)], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])
    algorithms = np.concatenate((_opt, algorithms))


# mean_revisit_time = np.zeros((len(job), len(alg_reprs)))
# for ii in range(len(job)):
#     for jj in range(len(alg_reprs)):
#         # mean_revisit_time[ii,jj] = np.mean(np.diff(np.append(0,Job_Revisit_Time[ii][jj]))) # Add 0 for cases where there is only 1 visit
#         if len(Job_Revisit_Time[ii][jj]) > 1:
#             mean_revisit_time[ii,jj] = np.mean(np.diff(Job_Revisit_Time[ii][jj])) # Add 0 for cases where there is only 1 visit
#         else:
#             mean_revisit_time[ii,jj] = np.nan

# mean_revisit_time = np.zeros((len(tasks_full), len(algorithms)))
mean_revisit_time = {}
mean_revisit_time_job_type = {}
Utility = {}
Penalty = {}
job_type_index = {}
for name, func, n_iter in algorithms:

    problem_gen = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail, record_revisit=True, scheduler=func)
    A = list(problem_gen(n_gen=1000))

    mean_revisit_time[name] = np.array([np.mean(np.diff(task.revisit_times)) for task in problem_gen.queue])


    # Display Revisit Rate by Job Type
    job_type = np.array([task.dwell_type for task in problem_gen.queue])
    job_unique = np.unique(job_type)
    index_pos_list = []
    UB_revisit_rate = np.array(np.zeros(len(job_unique))) # Calculate UB_revisit_rate here
    UB_job_type = np.array([1 / task.slope for task in problem_gen.queue])
    for count, type in enumerate(job_unique):
        job_type == type
        A = np.where(job_type == type)
        index_pos_list.append(A[:])
        UB_revisit_rate[count] = UB_job_type[index_pos_list[count][0][0]] # Only need one element, should all be the same
    job_type_index[name] = index_pos_list

    mean_revisit_time_job_type[name] = np.zeros(len(job_unique))
    Utility[name] = np.zeros(len(job_unique))
    Penalty[name] = np.zeros(len(job_unique))
    for ii in range(len(job_unique)):
        mean_revisit_time_job_type[name][ii] = np.mean(mean_revisit_time[name][index_pos_list[ii]])
        temp = UB_revisit_rate[ii] - mean_revisit_time[name][index_pos_list[ii]]
        Utility[name][ii] = np.sum(temp)
        Penalty[name][ii] = -np.sum(temp[temp < 0])

########################### Plotting ########################
color_scheme_bound = ['b', 'g', 'r', 'm', 'k']
color_scheme = [['crimson', 'green', 'navy', 'magenta', 'darkred'], ['cyan', 'teal','navy','aquamarine'], ['lime', 'yellowgreen', 'chartreuse', 'lightgreen'],
                ['magenta', 'maroon', 'voilet', 'fushia'], ['grey', 'chocolate', 'brown', 'beige'] ]

for count, alg in enumerate(algorithms):
    name, func, n_iter = alg
    # print(count, name, func, n_iter)

    plt.figure(200)
    plt.plot(job_unique, mean_revisit_time_job_type[name], label=name)
    plt.xlabel('Job Type')
    plt.ylabel('Mean Revisit Time')
    plt.title('Mean Revisit Time vs. Job Type')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.figure(201)
    plt.plot(job_unique, Penalty[name], label=name)
    plt.ylabel('Penalty')
    plt.xlabel('Job Type')
    plt.title('Penalty vs. Job Type')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.figure(100+count)
    plt.grid
    first_index = 0
    for ii in range(len(job_unique)):
        last_index = first_index + len(job_type_index[name][ii][0])
        y = np.arange(first_index,last_index)
        x = UB_revisit_rate[ii] * np.ones(np.shape(y))
        first_index = last_index + 1
        if ii == 0:
            plt.plot(x, y, color_scheme_bound[0], label='UB')
        else:
            plt.plot(x, y, color_scheme_bound[0])
        plt.text(x[0], y[0], 'Upper Bound: '+job_unique[ii])
        if ii == 0:
            plt.plot(mean_revisit_time[name][job_type_index[name][ii][0]], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])
        else:
            plt.plot(mean_revisit_time[name][job_type_index[name][ii][0]], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])

        # y2 = mean_revisit_time_job_type[name][ii] * np.ones(np.shape(x))
        # plt.plot(x, y2, color_scheme[ii])
        # plt.text(x[0], y2[0], 'Mean: '+ job_unique[ii])
    plt.ylabel('Sorted Job ID')
    plt.xlabel('Revisit Rate')
    plt.grid(True)
    plt.legend()
    plt.title(name + '\n Penalty = ' + str(np.sum(Penalty[name])))





    #
    # Alg_Penalty = np.sum(Penalty,axis=0)
    # plt.figure(202)
    # for jj in range(len(alg_reprs)):
    #     plt.plot(Alg_time[jj], Alg_Penalty[jj], marker='o', label=alg_reprs[jj])
    # plt.xlabel('Run Time (s)')
    # plt.ylabel('Penalty')
    # plt.legend()




    if 0: # Not sure why I was sorting the indices previously. Seems overly complicated. Remembering was having problems with python and finding the indicis of by job type. Trying something new above
        idx_sort = np.argsort(job_type)
        last_index = np.zeros(len(job_unique))
        for jj in range(len(job_unique)):
            # last_index[jj] = np.where(job_type == job_unique[jj])[0][-1]
            last_index[jj] = np.where(job_type[idx_sort] == job_unique[jj])[0][-1]

        # job_type = np.array([task.Type for task in job])  # Original ordering of job types. Needed for metrics later

        last_index = last_index.astype(int)
        first_index = np.append(0, last_index[0:len(job_unique)-1]+1 ).astype(int)
        UB_revisit_rate = np.array(np.zeros(len(first_index)))
        for jj in range(len(first_index)):
            UB_revisit_rate[jj] = UB_job_type[idx_sort[first_index[jj]]]
        # UB_revisit_rate = np.array([job[idx_sort[first_index[idx]]].t_drop for idx in range(N_job_types)])
        # desired_revisit_rate = job[idx_sort[first_index]].t_drop


        mean_revisit_time_job_type = np.zeros((len(job_unique), len(algorithms)))
        Utility = np.zeros((len(job_unique), len(algorithms)))
        Penalty = np.zeros((len(job_unique), len(algorithms)))
        for ii in range(len(job_unique)):
            for jj in range(len(algorithms)):
                idx_support = idx_sort[first_index[ii]:last_index[ii]]
                mean_revisit_time_job_type[ii,jj] = np.mean(mean_revisit_time[idx_support, jj])
                temp = UB_revisit_rate[ii] - mean_revisit_time[idx_support,jj]
                Utility[ii,jj] = np.sum(temp)
                Penalty[ii,jj] = -np.sum(temp[temp<0])

if train_RL_flag:
    plt.figure(1)
    plt.savefig('./Figures/' + filename_train + '_RL_TRAIN.eps', format='eps', dpi=600)

plt.figure(200)
plt.savefig('./Figures/' + filename_train + '_Mean_Revisit_Time.eps', format='eps', dpi=600)

plt.figure(201)
plt.savefig('./Figures/' + filename_train + '_Penalty.eps', format='eps', dpi=600)
# plt.figure('tra')

for count, alg in enumerate(algorithms):
    plt.figure(100+count)
    plt.savefig('./Figures/' + filename_train + alg[0] + '.eps', format='eps', dpi=600)

if train_SL_flag:
    plt.figure(num='training history')
    plt.savefig('./Figures/' + filename_train + '_training_history.eps', format='eps', dpi=600)


plt.show()

a = 1


# for task in problem_gen.queue:
#     print(task.revisit_times)

#
# mean_revisit_time = np.array([np.mean(np.diff(task.revisit_times)) for task in problem_gen.queue])
#
#
# plt.figure(100)
# plt.plot(mean_revisit_time)
# plt.xlabel('Job ID')
# plt.ylabel('Revisit Rate')
# plt.show()
# a = 1
#
#
#
# mean_revisit_time_job_type = np.zeros(len(job_unique))
# for ii in range(len(job_unique)):
#     mean_revisit_time_job_type[ii] = np.mean(mean_revisit_time[index_pos_list[ii]])



# idx_sort = np.argsort(job_type)
# last_index = np.zeros(len(job_unique))
# for jj in range(len(job_unique)):
#     # last_index[jj] = np.where(job_type == job_unique[jj])[0][-1]
#     last_index[jj] = np.where(job_type[idx_sort] == job_unique[jj])[0][-1]
#
# last_index = last_index.astype(int)
# first_index = np.append(0, last_index[0:N_job_types-1]+1 ).astype(int)
# UB_revisit_rate = np.array(np.zeros(len(first_index)))
# for jj in range(len(first_index)):
#     UB_revisit_rate[jj] = UB_job_type[idx_sort[first_index[jj]]]
# # UB_revisit_rate = np.array([job[idx_sort[first_index[idx]]].t_drop for idx in range(N_job_types)])
# # desired_revisit_rate = job[idx_sort[first_index]].t_drop
#
# mean_revisit_time_job_type = np.zeros((len(job_unique),len(alg_reprs)))
# Utility = np.zeros((len(job_unique),len(alg_reprs)))
# Penalty = np.zeros((len(job_unique),len(alg_reprs)))
# for ii in range(len(job_unique)):
#     for jj in range(len(alg_reprs)):
#         idx_support = idx_sort[first_index[ii]:last_index[ii]]
#         mean_revisit_time_job_type[ii,jj] = np.mean(mean_revisit_time[idx_support,jj])
#         temp = UB_revisit_rate[ii] - mean_revisit_time[idx_support,jj]
#         Utility[ii,jj] = np.sum(temp)
#         Penalty[ii,jj] = -np.sum(temp[temp<0])
#
#
#
#
# color_scheme_bound = ['b', 'g', 'r', 'm', 'k']
# color_scheme = [['crimson', 'green', 'navy', 'magenta', 'darkred'], ['cyan', 'teal','navy','aquamarine'], ['lime', 'yellowgreen', 'chartreuse', 'lightgreen'],
#                 ['magenta', 'maroon', 'voilet', 'fushia'], ['grey', 'chocolate', 'brown', 'beige'] ]
#
# for jj in range(len(alg_reprs)):
#     plt.figure(100+jj)
#     plt.grid
#     for ii in range(N_job_types):
#         y = np.arange(first_index[ii],last_index[ii]+1)
#         x = UB_revisit_rate[ii] * np.ones(np.shape(y))
#         if ii == 0:
#             plt.plot(x, y, color_scheme_bound[0], label='UB')
#         else:
#             plt.plot(x, y, color_scheme_bound[0])
#         plt.text(x[0], y[0], 'Upper-Bound: '+job_unique[ii])
#         if ii == 0:
#             plt.plot(mean_revisit_time[idx_sort[y], jj], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])
#         else:
#             plt.plot(mean_revisit_time[idx_sort[y], jj], y, color_scheme[0][ii], marker="o", linestyle='None', label=job_unique[ii])
#
#         # y2 = mean_revisit_time_job_type[ii,jj] * np.ones(np.shape(x))
#         # plt.plot(x, y2, color_scheme[ii])
#         # plt.text(x[0], y2[0], 'Mean: '+job_unique[ii])
#     plt.ylabel('Sorted Job ID')
#     plt.xlabel('Revisit Rate')
#     plt.grid(True)
#     plt.show()
#     plt.legend()
#     plt.title(alg_reprs[jj] + '\n Penalty = ' + str(np.sum(Penalty[:,jj])) )
#
#
# plt.figure(200)
# plt.plot(job_unique, mean_revisit_time_job_type, label=alg_reprs[jj])
# plt.xlabel('Job Type')
# plt.ylabel('Mean Revisit Time')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# plt.figure(201)
# for jj in range(len(alg_reprs)):
#     plt.plot(job_unique, Penalty[:,jj], label=alg_reprs[jj])
# plt.ylabel('Penalty')
# plt.xlabel('Job Type')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# Alg_Penalty = np.sum(Penalty,axis=0)
# plt.figure(202)
# for jj in range(len(alg_reprs)):
#     plt.plot(Alg_time[jj], Alg_Penalty[jj], marker='o', label=alg_reprs[jj])
# plt.xlabel('Run Time (s)')
# plt.ylabel('Penalty')
# plt.legend()



plt.show()
a = 1


# print('\nAvg. Performance\n' + 16 * '-')
# print(f"{'Algorithm:':<35}{'Loss:':<8}{'Runtime (s):':<10}")
# for name in algorithms['name']:
#     print(f"{name:<35}{l_ex_mean[name].mean():<8.2f}{t_run_mean[name].mean():<10.6f}")
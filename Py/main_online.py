import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import task_scheduling
from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms.free import earliest_release
from task_scheduling.learning import environments as envs
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.learning.RL_policy import ReinforcementLearningScheduler as RL_Scheduler
from task_scheduling.util.results import timing_wrapper, evaluate_algorithms, evaluate_algorithms_runtime
from task_scheduling.util.plot import scatter_loss_runtime_stats
from task_scheduling.tasks import check_task_types


def generate_data(create_data_flag=False, n_gen=None, n_tasks=None, n_track=None, n_ch=None):

    # create_data_flag = True

    if create_data_flag:

        # n_gen = 10000
        # n_tasks = 4  # Number of tasks to process at each iteration
        # n_track = 10
        # ch_avail = np.zeros(n_ch, dtype=np.float)
        tasks_full = task_gens.FlexDAR(n_track=n_track, rng=100).tasks_full

        # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
        # ch_avail = [0, 0]
        ch_avail = [0]*n_ch
        problem_gen = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)


        features = np.array([('duration', lambda task: task.duration, (0, 10)),
                             ('release time', lambda task: task.t_release, (0, 10)),
                             ('slope', lambda task: task.slope, (0, 10)),
                             ('drop time', lambda task: task.t_drop, (0, 10)),
                             ('drop loss', lambda task: task.l_drop, (0, 10)),
                             ],
                            dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

        # env_cls = envs.SeqTasking
        env_cls = envs.SeqTasking

        env_params = {'features': features,
                      'sort_func': None,
                      'time_shift': True,
                      'masking': True,
                      'action_type': 'int',
                      # 'action_type': 'any',
                      # 'seq_encoding': 'one-hot',
                      }

        env = env_cls(problem_gen, **env_params)

        filename = 'FlexDAR_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track) + \
                   '_' + str(n_gen)
        list(env.problem_gen(n_gen=n_gen, save=True, file=filename))




def test_rl_train():

    plot_hist_flag = False
    # n_gen = 100
    # n_train = np.array(n_gen*0.9, dtype=int)
    # n_eval = n_gen - n_train - 1
    n_train = 100
    n_eval = 200

    n_tasks = 5  # Number of tasks to process at each iteration
    n_track = 10
    n_track_eval = 11
    # ch_avail = np.zeros(2, dtype=np.float)
    tasks_full = task_gens.FlexDAR(n_track=n_track, rng=100).tasks_full

    # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
    n_ch = 1

    ch_avail = [0]*n_ch
    # n_ch = len(ch_avail)

    # Problem Generator
    # Use separate datasets for training and evaluation. Let Training dataset repeat for training.
    filename_train = 'FlexDAR_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track) + '_' + str(n_train)
    # Eval with 0 tracks for now
    filename_eval = 'FlexDAR_' + 'ch' + str(len(ch_avail)) + 't' + str(n_tasks) + '_track' + str(n_track_eval) + '_' + str(n_eval)
    filepath_train = './data/schedules/' + filename_train
    filepath_eval = './data/schedules/' + filename_eval
    if os.path.isfile(filepath_train):
        problem_gen = problem_gens.Dataset.load(file=filename_train, shuffle=False, rng=None, repeat=True)
    else:
        generate_data(create_data_flag=True, n_gen=n_train, n_tasks=n_tasks, n_track=n_track, n_ch=n_ch)
        problem_gen = problem_gens.Dataset.load(file=filename_train, shuffle=False, rng=None, repeat=True)

    if os.path.isfile(filepath_eval):
        problem_gen_eval = problem_gens.Dataset.load(file=filename_eval, shuffle=False, rng=None)
    else:
        generate_data(create_data_flag=True, n_gen=n_eval, n_tasks=n_tasks, n_track=n_track_eval, n_ch=n_ch)
        problem_gen_eval = problem_gens.Dataset.load(file=filename_eval, shuffle=False, rng=None)


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



    features = np.array([('duration', lambda task: task.duration, (0, 0.05)),
                         # ('release time', lambda task: task.t_release, (0, 1)),
                         # ('release time', lambda task: task.t_release - , (0, 10)),
                         ('slope', lambda task: task.slope, (0, 1)),
                         ('drop time', lambda task: task.t_drop, (0, 6)),
                         ('offset', lambda task: task.t_release - np.min(task.ch_avail), (-5, 0)),

                         # ('drop loss', lambda task: task.l_drop, (0, 10)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

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

    dqn_agent = RL_Scheduler.train_from_gen(problem_gen, env_cls, env_params,
                                            model_cls='DQN_LN', model_params={'verbose': 1}, n_episodes=n_train * 100,
                                            # model_cls='CNN', model_params={'verbose': 1}, n_episodes=n_train*100,
                                            # model_cls='DQN', model_params={'verbose': 1}, n_episodes=n_train * 100,
                                            save=False, save_path='./')

    algorithms = np.array([
        # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
        # ('Random', algs_base.random_sequencer, 20),
        ('ERT', earliest_release, 1),
        # ('MCTS', partial(algs_base.mcts, n_mc=100, verbose=False), 5),
        ('DQN Agent', dqn_agent, 1),
        # ('DNN Policy', policy_model, 5),
    ], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

    # l_ex_iter, t_run_iter, l_ex_mean, t_run_mean, l_ex_mean_norm = evaluate_algorithms(algorithms, problem_gen_eval,
    #                                                                                    n_gen=n_eval, solve=True,
    #                                                                                    verbose=2, plotting=1,
    #                                                                                    save=False, save_path=None)

    l_ex_mean, t_run_mean = evaluate_algorithms(algorithms, problem_gen_eval, n_gen=n_eval, solve=True, verbose=2,
                                                plotting=1, data_path=None)

    scatter_loss_runtime_stats(t_run_mean, l_ex_mean, ax=None, ax_kwargs=None)

    plt.show()
    a = 1

    print('\nAvg. Performance\n' + 16 * '-')
    print(f"{'Algorithm:':<35}{'Loss:':<8}{'Runtime (s):':<10}")
    for name in algorithms['name']:
        print(f"{name:<35}{l_ex_mean[name].mean():<8.2f}{t_run_mean[name].mean():<10.6f}")




    # maxTime = 10
    # n_step = np.int(np.floor(maxTime / env.problem_gen.RP))
    # for ii in range(n_step):
    #     env.problem_gen.clock = ii * env.problem_gen.RP
    #     # print(", ".join([f"{task.t_release:.2f}" for task in env.problem_gen.tasks]))
    #
    #     if np.min(ch_avail) > env.problem_gen.clock:
    #         continue
    #
    #     if env.problem_gen.clock % env.problem_gen.RP * 1 == 0:
    #         print('time =', env.problem_gen.clock)
    #
    #     env.problem_gen.summary()
    #     env.problem_gen.reprioritize()
    #     env.problem_gen.summary()
    #
    #     temp = list(env.problem_gen(1))
    #     tasks = temp[0][0]
    #     env.problem_gen.summary()
    #
    #     # t_ex, ch_ex = earliest_release(tasks, ch_avail)
    #     t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)
    #
    #     # TODO: use t_run to check validity of t_ex
    #     # t_ex = np.max([t_ex, [t_run for _ in range(len(t_ex))]], axis=0)
    #
    #     # obs, reward, done, info = env.step()
    #
    #     # done = False
    #     # while not done:
    #     #     obs, reward, done, info = env.step(action)
    #
    #     env.problem_gen.updateFlexDAR(tasks, t_ex, ch_ex)
    #     # q.summary()


# def test_queue():
#
#     n_tasks = 8  # Number of tasks to process at each iteration
#     n_track = 10
#     ch_avail = np.zeros(2, dtype=np.float)
#     tasks_full = task_gens.FlexDAR(n_track=n_track).tasks_full
#
#     # tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
#     # tasks_full = task_gens.FlexDAR(n_track=10)()
#
#     # df = pd.DataFrame({name: [getattr(task, name) for task in tasks_full]
#     #                    for name in tasks_full._cls_task.param_names})
#     # print(df)
#
#
#     # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
#     ch_avail = [0, 0]
#     # Problem Generator
#     q = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)
#
#     features = np.array([('duration', lambda task: task.duration, (0, 10)),
#                          ('release time', lambda task: task.t_release, (0, 10)),
#                          ('slope', lambda task: task.slope, (0, 10)),
#                          ('drop time', lambda task: task.t_drop, (0, 10)),
#                          ('drop loss', lambda task: task.l_drop, (0, 10)),
#                          ],
#                         dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
#
#     env_cls = envs.SeqTasking
#     # env_cls = envs.StepTasking
#
#     env_params = {'features': features,
#                   'sort_func': None,
#                   'time_shift': True,
#                   'masking': True,
#                   'action_type': 'int',
#                   # 'action_type': 'any',
#                   # 'seq_encoding': 'one-hot',
#                   }
#
#     env = env_cls(q, **env_params)
#     env.problem_gen(1, solve=False)
#
#     env.reset()
#
#     dqn_agent = RL_Scheduler.train_from_gen(q, env_cls, env_params,
#                                             model_cls='DQN', model_params={'verbose': 1}, n_episodes=10000,
#                                             save=False, save_path=None)
#
#
#
#     maxTime = 10
#     n_step = np.int(np.floor(maxTime / env.problem_gen.RP))
#     for ii in range(n_step):
#         env.problem_gen.clock = ii*env.problem_gen.RP
#         # print(", ".join([f"{task.t_release:.2f}" for task in env.problem_gen.tasks]))
#
#         if np.min(ch_avail) > env.problem_gen.clock:
#             continue
#
#         if env.problem_gen.clock % env.problem_gen.RP * 1 == 0:
#             print('time =', env.problem_gen.clock)
#
#         env.problem_gen.summary()
#         env.problem_gen.reprioritize()
#         env.problem_gen.summary()
#
#         temp = list(env.problem_gen(1))
#         tasks = temp[0][0]
#         env.problem_gen.summary()
#
#         # t_ex, ch_ex = earliest_release(tasks, ch_avail)
#         t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)
#
#         # TODO: use t_run to check validity of t_ex
#         # t_ex = np.max([t_ex, [t_run for _ in range(len(t_ex))]], axis=0)
#
#         # obs, reward, done, info = env.step()
#
#         # done = False
#         # while not done:
#         #     obs, reward, done, info = env.step(action)
#
#
#         env.problem_gen.updateFlexDAR(tasks, t_ex, ch_ex)
#         # q.summary()
#
#
# def test_queue_env():
#     tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
#     # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
#     ch_avail = [0, .1]
#
#     problem_gen = problem_gens.Queue(2, tasks_full, ch_avail)
#
#     features = np.array([('duration', lambda task: task.duration, (0, 10)),
#                          ('release time', lambda task: task.t_release, (0, 10)),
#                          ('slope', lambda task: task.slope, (0, 10)),
#                          ('drop time', lambda task: task.t_drop, (0, 10)),
#                          ('drop loss', lambda task: task.l_drop, (0, 10)),
#                          ],
#                         dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
#
#     # env_cls = envs.SeqTasking
#     env_cls = envs.StepTasking
#
#     env_params = {'features': features,
#                   'sort_func': None,
#                   'time_shift': True,
#                   'masking': True,
#                   # 'action_type': 'seq',
#                   'action_type': 'any',
#                   'seq_encoding': 'one-hot',
#                   }
#
#     env = env_cls(problem_gen, **env_params)
#
#     for _ in range(2):
#         env.problem_gen.summary()
#
#         obs = env.reset()
#         env.problem_gen.summary()
#
#         tasks, ch_avail = env.tasks, env.ch_avail
#         t_ex, ch_ex = earliest_release(tasks, ch_avail)
#
#         env.problem_gen.update(tasks, t_ex, ch_ex)      # TODO?
#         env.problem_gen.summary()
#
#         seq = np.argsort(t_ex)
#         for n in seq:
#             obs, reward, done, info = env.step(n)
#
#         # done = False
#         # while not done:
#         #     obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    test_rl_train()
    # test_queue()
    # test_queue_env()

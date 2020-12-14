import numpy as np

import task_scheduling
from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms.base import earliest_release
from task_scheduling.learning import environments as envs
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.util.results import timing_wrapper
from task_scheduling.learning.RL_policy import ReinforcementLearningScheduler as RL_Scheduler
from task_scheduling.util.results import evaluate_algorithms, evaluate_algorithms_runtime


ch_avail = np.zeros(2, dtype=np.float)
# tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
tasks_full = [task_scheduling.tasks.ReluDrop(1, 0, slope, 5, 10) for slope in np.arange(1, 1.4, .1)]
# tasks_full = [task_scheduling.tasks.ReluDropRadar.search(0.018, 'AHS') for _ in range(4)]




# def get_tasks(tasks_):
#     tasks_sort = sorted(tasks_, key=lambda task_: task_.t_release)
#     return tasks_sort[:2]

def priority(task_):
    return -task_.t_release


# t_clock = 0.
# t_del = 0.01
loss_full = 0.
for __ in range(100):
    # tasks = get_tasks(tasks_full)
    tasks_full.sort(key=priority)
    tasks = tasks_full[-2:]

    t_ex, ch_ex = earliest_release(tasks, ch_avail)
    # t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)

    # Scheduled task updates
    for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
        loss_full += task(t_ex_i)

        task.t_release = t_ex_i + task.duration
        ch_avail[ch_ex_i] = max(ch_avail[ch_ex_i], task.t_release)      # TODO: get from TreeNode?

    # TODO: effectively jumps sim time to ch_avail_min

    # Dropped task updates
    ch_avail_min = min(ch_avail)
    for task in tasks_full:
        while task.t_release + task.t_drop < ch_avail_min:      # absolute drop time
            loss_full += task.l_drop        # add drop loss
            task.t_release += task.t_drop   # increment release time

    # t_clock += t_del


def test_env():
    n_tasks = 4  # Number of tasks to process at each iteration
    n_track = 10
    ch_avail = np.zeros(2, dtype=np.float)
    tasks_full = task_gens.FlexDAR(n_track=n_track).tasks_full


    # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
    ch_avail = [0, 0]
    # Problem Generator
    q = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)

    features = np.array([('duration', lambda task: task.duration, (0, 10)),
                         ('release time', lambda task: task.t_release, (0, 10)),
                         ('slope', lambda task: task.slope, (0, 10)),
                         ('drop time', lambda task: task.t_drop, (0, 10)),
                         ('drop loss', lambda task: task.l_drop, (0, 10)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    # env_cls = envs.SeqTasking
    env_cls = envs.SeqTasking

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': None,
                  'masking': True,
                  'action_type': 'int',
                  # 'action_type': 'any',
                  # 'seq_encoding': 'one-hot',
                  }

    env = env_cls(q, **env_params)
    # env.problem_gen(1, solve=False)
    # env.reset()
    # for __ in range(10):
    #     (tasks, ch_avail), = env.problem_gen(1, solve=False)



    dqn_agent = RL_Scheduler.train_from_gen(q, env_cls, env_params,
                                            model_cls='DQN', model_params={'verbose': 1}, n_episodes=1000,
                                            save=False, save_path=None)

    algorithms = np.array([
        # ('B&B sort', sort_wrapper(partial(branch_bound, verbose=False), 't_release'), 1),
        # ('Random', algs_base.random_sequencer, 20),
        ('ERT', earliest_release, 1),
        # ('MCTS', partial(algs_base.mcts, n_mc=100, verbose=False), 5),
        ('DQN Agent', dqn_agent, 5),
        # ('DNN Policy', policy_model, 5),
    ], dtype=[('name', '<U16'), ('func', np.object), ('n_iter', np.int)])

    l_ex_iter, t_run_iter = evaluate_algorithms(algorithms, q, n_gen=100, solve=True,
                                                verbose=2, plotting=1, save=True, file=None)


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


def test_queue():

    n_tasks = 8  # Number of tasks to process at each iteration
    n_track = 10
    ch_avail = np.zeros(2, dtype=np.float)
    tasks_full = task_gens.FlexDAR(n_track=n_track).tasks_full

    # tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
    # tasks_full = task_gens.FlexDAR(n_track=10)()

    # df = pd.DataFrame({name: [getattr(task, name) for task in tasks_full]
    #                    for name in tasks_full._cls_task.param_names})
    # print(df)


    # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
    ch_avail = [0, 0]
    # Problem Generator
    q = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)

    features = np.array([('duration', lambda task: task.duration, (0, 10)),
                         ('release time', lambda task: task.t_release, (0, 10)),
                         ('slope', lambda task: task.slope, (0, 10)),
                         ('drop time', lambda task: task.t_drop, (0, 10)),
                         ('drop loss', lambda task: task.l_drop, (0, 10)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    # env_cls = envs.SeqTasking
    env_cls = envs.SeqTasking

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': None,
                  'masking': True,
                  'action_type': 'int',
                  # 'action_type': 'any',
                  # 'seq_encoding': 'one-hot',
                  }

    env = env_cls(q, **env_params)
    env.problem_gen(1, solve=False)

    env.reset()

    dqn_agent = RL_Scheduler.train_from_gen(q, env_cls, env_params,
                                            model_cls='DQN', model_params={'verbose': 1}, n_episodes=10000,
                                            save=False, save_path=None)



    maxTime = 10
    n_step = np.int(np.floor(maxTime/env.problem_gen.RP))
    for ii in range(n_step):
        env.problem_gen.clock = ii*env.problem_gen.RP
        # print(", ".join([f"{task.t_release:.2f}" for task in env.problem_gen.tasks]))

        if np.min(ch_avail) > env.problem_gen.clock:
            continue

        if env.problem_gen.clock % env.problem_gen.RP * 1 == 0:
            print('time =', env.problem_gen.clock)

        env.problem_gen.summary()
        env.problem_gen.reprioritize()
        env.problem_gen.summary()

        temp = list(env.problem_gen(1))
        tasks = temp[0][0]
        env.problem_gen.summary()

        # t_ex, ch_ex = earliest_release(tasks, ch_avail)
        t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)

        # TODO: use t_run to check validity of t_ex
        # t_ex = np.max([t_ex, [t_run for _ in range(len(t_ex))]], axis=0)

        # obs, reward, done, info = env.step()

        # done = False
        # while not done:
        #     obs, reward, done, info = env.step(action)


        env.problem_gen.updateFlexDAR(tasks, t_ex, ch_ex)
        # q.summary()




def test_queue_env():
    tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
    # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
    ch_avail = [0, .1]

    problem_gen = problem_gens.Queue(2, tasks_full, ch_avail)

    features = np.array([('duration', lambda task: task.duration, (0, 10)),
                         ('release time', lambda task: task.t_release, (0, 10)),
                         ('slope', lambda task: task.slope, (0, 10)),
                         ('drop time', lambda task: task.t_drop, (0, 10)),
                         ('drop loss', lambda task: task.l_drop, (0, 10)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    # env_cls = envs.SeqTasking
    env_cls = envs.StepTasking

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': None,
                  'masking': True,
                  # 'action_type': 'seq',
                  'action_type': 'any',
                  'seq_encoding': 'one-hot',
                  }

    env = env_cls(problem_gen, **env_params)

    for _ in range(2):
        env.problem_gen.summary()

        obs = env.reset()
        env.problem_gen.summary()

        tasks, ch_avail = env.tasks, env.ch_avail
        t_ex, ch_ex = earliest_release(tasks, ch_avail)

        env.problem_gen.update(tasks, t_ex, ch_ex)      # TODO?
        env.problem_gen.summary()

        seq = np.argsort(t_ex)
        for n in seq:
            obs, reward, done, info = env.step(n)

        # done = False
        # while not done:
        #     obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    test_env()
    # test_queue()
    # test_queue_env()

import numpy as np

from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms.base import earliest_release
from task_scheduling.learning import environments as envs
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.util.results import timing_wrapper


ch_avail = np.zeros(2, dtype=np.float)
tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))


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

    for task, t_ex_i, ch_ex_i in zip(tasks, t_ex, ch_ex):
        loss_full += task(t_ex_i)

        task.t_release = t_ex_i + task.duration
        ch_avail[ch_ex_i] = max(ch_avail[ch_ex_i], task.t_release)      # TODO: get from TreeNode?

    # TODO: update ALL tasks based on new time. Drops, loss incurred...
    ch_avail_min = min(ch_avail)
    for task in tasks_full:
        while task.t_release + task.t_drop < ch_avail_min:      # absolute drop time
            loss_full += task.l_drop        # add drop loss
            task.t_release += task.t_drop   # increment release time
            # task.t_release = ch_avail_min


    # t_clock += t_del



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
    q = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)
    maxTime = 10
    n_step = np.int(np.floor(maxTime/q.RP))
    for ii in range(n_step):
        q.clock = ii*q.RP
        # print(", ".join([f"{task.t_release:.2f}" for task in q.tasks]))

        if np.min(ch_avail) > q.clock:
            continue

        if q.clock % q.RP * 1 == 0:
            print('time =', q.clock)

        q.summary()
        q.reprioritize()
        q.summary()

        temp = list(q(1))
        tasks = temp[0][0]
        q.summary()

        # t_ex, ch_ex = earliest_release(tasks, ch_avail)
        t_ex, ch_ex, t_run = timing_wrapper(earliest_release)(tasks, ch_avail)

        # TODO: use t_run to check validity of t_ex
        # t_ex = np.max([t_ex, [t_run for _ in range(len(t_ex))]], axis=0)

        q.updateFlexDAR(tasks, t_ex, ch_ex)
        q.summary()

if 0:
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
    test_queue()
    # test_queue_env()

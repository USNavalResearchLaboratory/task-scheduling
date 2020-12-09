import numpy as np

from task_scheduling.generators import (tasks as task_gens, channel_availabilities as ch_gens,
                                        scheduling_problems as problem_gens)
from task_scheduling.algorithms.base import earliest_release
from task_scheduling.learning import environments as envs
from task_scheduling.tree_search import TreeNodeShift


def test_queue():



    n_tasks = 8

    # tasks_full = list(task_gens.ContinuousUniformIID.relu_drop()(4))
    tasks_full = task_gens.FlexDAR(n_track=10).tasks_full

    # df = pd.DataFrame({name: [getattr(task, name) for task in tasks_full]
    #                    for name in tasks_full._cls_task.param_names})
    # print(df)

    # ch_avail = list(ch_gens.UniformIID((0, 0))(2))
    ch_avail = [0, 0]

    # RP = 40*1e-3 # Resource period in seconds, set to 40 ms by default
    # clock = 0
    q = problem_gens.QueueFlexDAR(n_tasks, tasks_full, ch_avail)
    for _ in range(4):
        # print(", ".join([f"{task.t_release:.2f}" for task in q.tasks]))
        q.summary()

        q.reprioritize()
        q.summary()

        tasks = list(q(2))
        q.summary()

        t_ex, ch_ex = earliest_release(tasks, ch_avail)

        q.update(tasks, t_ex, ch_ex)
        q.summary()


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
    # test_queue()
    test_queue_env()

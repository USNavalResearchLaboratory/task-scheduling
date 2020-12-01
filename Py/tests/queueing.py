from task_scheduling.generators import tasks as task_gens, channel_availabilities as ch_gens
from task_scheduling.algorithms.base import earliest_release


# TODO: make task list summary util func!

def test_queue():

    tasks_master = list(task_gens.ContinuousUniformIID.relu_drop()(4))
    ch_avail = list(ch_gens.UniformIID((0, 0))(2))

    q = task_gens.Queue(tasks_master)
    for _ in range(1):
        # print(", ".join([f"{task.t_release:.2f}" for task in q.tasks]))
        tasks = list(q(2))
        # print(", ".join([f"{task.t_release:.2f}" for task in q.tasks]))
        t_ex, ch_ex = earliest_release(tasks, ch_avail)
        # TODO: calculate/return channel avails?
        q.update(tasks, t_ex)
        # print(", ".join([f"{task.t_release:.2f}" for task in q.tasks]))

        # tasks_scheduled, tasks_unscheduled = None, None
        # q.update(tasks_scheduled, solution)
        # q.add_tasks(tasks_unscheduled)


if __name__ == '__main__':
    test_queue()

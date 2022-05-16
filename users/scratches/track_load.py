from functools import partial
from matplotlib import pyplot as plt
from operator import attrgetter

from task_scheduling import algorithms
from task_scheduling.tasks import Linear, LinearDrop, LinearLinear
from task_scheduling.util import summarize_tasks, check_schedule, evaluate_schedule, plot_losses_and_schedule

plt.style.use(r'/images/style.mplstyle')
plt.rc('text', usetex=False)
# plt.rc('figure', autolayout=True)


n_search = 4
tasks_search = [Linear(.036, 0., slope=1., name=f"s_{i}") for i in range(n_search)]

n_track = 4
tasks_track = [LinearDrop(.018, 0., slope=2, t_drop=.06, l_drop=None, name=f"t_{i}") for i in range(n_track)]

tasks = tasks_search + tasks_track
ch_avail = [0.]

# print(summarize_tasks(tasks))
# plot_task_losses(tasks)

print(n_search * 1. * .018, tasks_track[0].l_drop)


# Define and assess algorithms
algorithms = dict(
    BB=partial(algorithms.branch_bound_priority, verbose=True),
    Sort=partial(algorithms.priority_sorter, func=attrgetter('slope'), reverse=True),
    # ERT=algorithms.earliest_release,
    # Random=algorithms.random_sequencer,
)

for name, algorithm in algorithms.items():
    sch = algorithm(tasks, ch_avail)

    check_schedule(tasks, sch)
    loss = evaluate_schedule(tasks, sch)
    plot_losses_and_schedule(tasks, sch, ch_avail, loss, name, fig_kwargs=dict(figsize=[6.4, 4.8]))
    print(f"{name}: {loss}")

from matplotlib import pyplot as plt

from task_scheduling import algorithms
from task_scheduling.generators import tasks as task_gens
from task_scheduling.util import summarize_tasks, plot_task_losses, plot_schedule, check_schedule, evaluate_schedule

plt.style.use('../images/style.mplstyle')
seed = 12345

# Define scheduling problem
task_gen = task_gens.ContinuousUniformIID.linear_drop(rng=seed)

tasks = list(task_gen(8))
ch_avail = [0., 0.5]

print(summarize_tasks(tasks))
plot_task_losses(tasks)


# Define and assess algorithms
algorithms = dict(
    Optimal=algorithms.branch_bound_priority,
    Random=algorithms.random_sequencer,
)

__, axes = plt.subplots(len(algorithms))
for (name, algorithm), ax in zip(algorithms.items(), axes):
    sch = algorithm(tasks, ch_avail)

    check_schedule(tasks, sch)
    loss = evaluate_schedule(tasks, sch)
    plot_schedule(tasks, sch, loss=loss, name=name, ax=ax)

plt.show()

from matplotlib import pyplot as plt

from task_scheduling import algorithms
from task_scheduling.generators import tasks as task_gens
from task_scheduling.util import (
    check_schedule,
    evaluate_schedule,
    plot_schedule,
    plot_task_losses,
    summarize_tasks,
)

seed = 12345

# Define scheduling problem
task_gen = task_gens.ContinuousUniformIID.linear_drop(rng=seed)

tasks = list(task_gen(8))
ch_avail = [0.0, 0.5]

print(summarize_tasks(tasks))
plot_task_losses(tasks)
plt.savefig("Tasks.png")


# Define and assess algorithms
algorithms = dict(
    Optimal=algorithms.branch_bound_priority,
    Random=algorithms.random_sequencer,
)

for name, algorithm in algorithms.items():
    sch = algorithm(tasks, ch_avail)

    check_schedule(tasks, sch)
    loss = evaluate_schedule(tasks, sch)
    plot_schedule(tasks, sch, loss=loss, name=name)
    plt.savefig(f"{name}.png")

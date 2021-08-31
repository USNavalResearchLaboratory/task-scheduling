from matplotlib import pyplot as plt

from task_scheduling import algorithms
from task_scheduling.generators import tasks as task_gens
from task_scheduling.util import summarize_tasks, plot_task_losses, plot_schedule, check_schedule, evaluate_schedule

plt.style.use('seaborn')

seed = 12345


#%% Define scheduling problem
task_gen = task_gens.ContinuousUniformIID.relu_drop(rng=seed)

tasks = list(task_gen(8))
ch_avail = [0., 0.5]

print(summarize_tasks(tasks))
plot_task_losses(tasks)


#%% Define and assess algorithms
algorithms = [
    algorithms.branch_bound_priority,
    algorithms.random_sequencer,
]

__, axes = plt.subplots(len(algorithms))
for algorithm, ax in zip(algorithms, axes):
    t_ex, ch_ex = algorithm(tasks, ch_avail)

    check_schedule(tasks, t_ex, ch_ex)
    loss = evaluate_schedule(tasks, t_ex)
    plot_schedule(tasks, t_ex, ch_ex, l_ex=loss, ax=ax)

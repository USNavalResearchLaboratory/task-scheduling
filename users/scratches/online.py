from copy import deepcopy
from functools import partial
from operator import attrgetter

import numpy as np
from matplotlib import pyplot as plt

from task_scheduling import algorithms
from task_scheduling.tasks import Exponential, Linear, LinearDrop
from task_scheduling.util import (
    check_schedule,
    evaluate_schedule,
    plot_losses_and_schedule,
    summarize_tasks,
)

plt.style.use("images/style.mplstyle")
plt.rc("text", usetex=False)


n_block = 3

ch_avail_init = [0.0]

# tasks_search = [Linear(.036, 0., slope=1., name=f"s_{i}") for i in range(4)]
tasks_search = [Exponential(0.036, 0.0, a=1.0, b=100, name=f"s_{i}") for i in range(4)]

tasks_track = [
    LinearDrop(0.018, 0.0, slope=2, t_drop=0.06, l_drop=None, name=f"t_{i}") for i in range(0)
]

tasks_init = tasks_search + tasks_track


algorithms = dict(
    BB=partial(algorithms.branch_bound_priority, verbose=True),
    # Sort=partial(algorithms.priority_sorter, func=attrgetter('slope'), reverse=True),
    # ERT=algorithms.earliest_release,
    # Random=algorithms.random_sequencer,
)

for name, algorithm in algorithms.items():
    tasks = deepcopy(tasks_init)
    ch_avail = deepcopy(ch_avail_init)
    losses = []

    for i_b in range(n_block):
        sch = algorithm(tasks, ch_avail)

        check_schedule(tasks, sch)
        loss = evaluate_schedule(tasks, sch)
        plot_losses_and_schedule(
            tasks,
            sch,
            len(ch_avail),
            loss,
            f"{name}_{i_b}",
            fig_kwargs=dict(figsize=[6.4, 4.8]),
        )

        # Update channel availability
        for i_c in range(len(ch_avail)):
            ch_avail[i_c] = max(
                ch_avail[i_c],
                *(t + task.duration for task, (t, c) in zip(tasks, sch) if c == i_c),
            )

        # Update tasks
        for task, (t, c) in zip(tasks, sch):
            task.t_release = t + task.duration

            # ch_avail_min = min(ch_avail)
            # loss += task.shift_origin(ch_avail_min)
            # task.t_release = ch_avail_min

        losses.append(loss)

    loss_total = sum(losses)

    print(losses)
    print(loss_total)

plt.show()

# Task Scheduling
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811866.svg)](https://doi.org/10.5281/zenodo.6811866)

This Python package provides a framework for implementing task scheduling algorithms and assessing their performance. It includes traditional schedulers as well as both supervised and reinforcement learning schedulers.

## Installation
The `task_scheduling` package is developed for [Python 3.9](https://www.python.org/downloads/). Best practice is to first create a [virtual environment](https://docs.python.org/3/tutorial/venv.html). The package can then be installed directly from GitHub using
```
pip install git+https://github.com/USNavalResearchLaboratory/task-scheduling.git#egg=task_scheduling
```
To install a specific version or branch, specify a ref as detailed [here](https://pip.pypa.io/en/stable/topics/vcs-support/). Alternatively, the package can be installed locally using
```
git clone https://github.com/USNavalResearchLaboratory/task-scheduling.git
pip install task-scheduling/
```
Note that with both methods, the [editable option](https://pip.pypa.io/en/stable/cli/pip_install/) can be included to track any package modifications.


## Documentation
Package API documentation is available [here](https://usnavalresearchlaboratory.github.io/task-scheduling/).

Alternatively, the docs can be generated using the `sphinx` package and the `sphinx-rtd-theme`, both installable using `pip`. To build the HTML documentation, run `make html` from the `docs/` folder; the top level document will be `docs/build/html/index.html`

## Development
`task-scheduling` is being developed for the Cognitive Resource Management project @ U.S. Naval Research Laboratory. It is maintained by [Paul Rademacher](https://github.com/rademacher-p) and NRL Radar Division. For contribution and/or collaboration, please [contact us](mailto:paul.rademacher@nrl.navy.mil,kevin.wagner@nrl.navy.mil).

## Quickstart

### Tasks
Task objects must expose two attributes:
- `duration` - the time required to execute a task
- `t_release` - the earliest time at which a task may be executed

The tasks must implement a `__call__` method that provides a monotonic non-decreasing loss function quantifying the penalty for delayed execution.

One generic built-in task type is provided: `task_scheduling.tasks.PiecewiseLinear`; also included are special subclasses `Linear` and `LinearDrop`. The latter type is so-named because it implements a loss function that increases linearly from zero according to a positive parameter `slope` and then after a "drop" time `t_drop`, a large constant loss `l_drop` is incurred. Example loss functions are shown below.

![Task loss functions](images/ex_tasks.png)


### Algorithms
The task scheduling problem is defined using two variables:
- `tasks`, an array of task objects
- `ch_avail`, an array of channel availability times

and the scheduling solution is defined using a [NumPy structured array](https://numpy.org/doc/stable/user/basics.rec.html) `sch` of length `len(tasks)` with two fields:
- `t`, execution times (`float`)
- `c`, execution channels (`int`, in `range(len(ch_avail))`)

To be valid (as assessed using `util.check_schedule`), the execution times, execution channels, and task durations must be such that no two tasks on the same channel are executing at the same time.

Each algorithm is a Python `callable` implementing the same API; it takes two leading positional arguments `tasks` and `ch_avail` and returns the schedule array `sch`. An example schedule is shown below.

![Task schedule](images/ex_schedule.png)

#### Traditional schedulers
A variety of classic schedulers are provided in the `algorithms` subpackage:

- Optimal
  - Branch and Bound (B&B)
  - Brute force
- Searches
  - Monte Carlo Tree Search (MCTS)
- Fast heuristics
  - Earliest release time
  - Earliest drop time
  - Random sequencer

#### Learning schedulers
Traditional schedulers typically suffer from one of two drawbacks: high computational load or poor performance. New algorithms that learn from related problems may generalize well, finding near-optimal schedules in a shorter, more practical amount of runtime.

The `mdp` subpackage implements the scheduling problem as a Markov decision process for supervised and reinforcement learning. Scheduling environments are provided in `mdp.environments`, following the [OpenAI Gym](https://gym.openai.com/) API. The primary `Env` class is `Index`; this environment uses single task assignments as actions and converts the scheduling problem (tasks and channel availabilities) into observed states, including the status of each task. The `mdp.supervised` submodule provides scheduler objects that use policy networks (implemented with [PyTorch](https://pytorch.org/)) to learn from these environments. The `mdp.reinforcement` submodule provides schedulers that implement and use agents from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/); also included are special policies for Actor-Critic and DQN that enforce valid actions throughout the MDP episode.

### Evaluation
The primary metrics used to evaluate a scheduling algorithm are its achieved loss and its runtime. The `util.evaluate_schedule` function calculates the total loss; the `util.eval_wrapper` function allows any scheduler to be timed and assessed.

While these functions can be invoked directly, the package provides a number of convenient functions in the `results` submodule that automate this functionality, allow printing result tables to file, provide visuals, etc. The functions `evaluate_algorithms_single` and `evaluate_algorithms_gen` assess for single scheduling problems and across a set of generated problems, respectively. The function `evaluate_algorithms_train` adds an additional level of Monte Carlo iteration by re-training any learning schedulers a number of times.

Example result outputs are shown below for `evaluate_algorithms_gen`. The first scatter plot shows raw loss-runtime pairs, one for each scheduling problem; the second plot shows excess loss values relative to optimal. The Markdown format table provides the average values.

![Loss-runtime results](./images/ex_scatter.png)
![Loss-runtime results](./images/ex_scatter_relative.png)

```markdown
|            | Excess Loss (%) | Loss    | Runtime (ms) |
| ---------- | --------------- | ------- | ------------ |
| BB Optimal | 0.000           | 182.440 | 207.535      |
| Random     | 33.041          | 242.720 | 0.274        |
| ERT        | 28.525          | 234.481 | 0.260        |
| MCTS       | 13.284          | 206.675 | 6.340        |
| Lit Policy | 4.188           | 190.079 | 4.438        |
```

## Examples

### Basics (`examples/basics.py`)
The following example shows a single scheduling problem and solution. Tasks are created using one of the provided generators and a number of algorithms provide scheduling solutions. Built-in utilities help visualize the both the problem and the various solutions.

```python
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

```

### Policy learning and Monte Carlo assessment (`examples/learning.py`)
The following example demonstrates the definition of a scheduling problem generator, the creation of a learning scheduler using PyTorch Lightning, and the comparison of traditional vs. learning schedulers using Monte Carlo evaluation.

Note that the problem generator is used to instantiate the Environment, which is used to create and train the supervised learning policy. Also, note the structure of the `algorithms` array; each algorithm is has a name, a `callable`, and a number of iterations to perform per problem (averaging is best-practice for stochastic schedulers).

```python
from functools import partial

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from torch import nn

from task_scheduling.algorithms import earliest_release, mcts, random_sequencer
from task_scheduling.generators import problems as problem_gens
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.reinforcement import (
    MultiExtractor,
    StableBaselinesScheduler,
    ValidActorCriticPolicy,
)
from task_scheduling.mdp.supervised import LitScheduler
from task_scheduling.results import evaluate_algorithms_train

np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.3f}".format
seed = 12345

if seed is not None:
    seed_everything(seed)


# Define scheduling problem and algorithms
problem_gen = problem_gens.Random.continuous_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Dataset.load('data/continuous_linear_drop_c1t8', repeat=True)

env = Index(problem_gen, sort_func="t_release", reform=True)


learn_params = {
    "frac_val": 0.3,
    "max_epochs": 2000,
    "dl_kwargs": dict(batch_size=160, shuffle=True),
}
trainer_kwargs = {
    "logger": False,
    "enable_checkpointing": False,
    "callbacks": EarlyStopping("val_loss", patience=100),
    "accelerator": "auto",
}
lit_scheduler = LitScheduler.mlp(
    env,
    hidden_sizes_joint=[400],
    model_kwargs={"optim_params": {"lr": 1e-4}},
    trainer_kwargs=trainer_kwargs,
    learn_params=learn_params,
)


learn_params_sb = {
    "frac_val": 0.3,
    "max_epochs": 2000,
    "eval_callback_kwargs": dict(
        callback_after_eval=StopTrainingOnNoModelImprovement(1000, min_evals=0, verbose=1),
        n_eval_episodes=100,
        eval_freq=1000,
        verbose=1,
    ),
}
sb_model_kwargs = dict(
    policy=ValidActorCriticPolicy,
    policy_kwargs=dict(
        features_extractor_class=MultiExtractor.mlp,
        features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[]),
        net_arch=[400],
        activation_fn=nn.ReLU,
        normalize_images=False,
        infer_valid_mask=env.infer_valid_mask,
    ),
)
sb_scheduler = StableBaselinesScheduler.make_model(env, "PPO", sb_model_kwargs, learn_params_sb)


algorithms = np.array(
    [
        ("Random", random_sequencer, 10),
        ("ERT", earliest_release, 10),
        ("MCTS", partial(mcts, max_rollouts=10, c_explore=0.05, th_visit=5), 10),
        ("SL Policy", lit_scheduler, 10),
        ("RL Agent", sb_scheduler, 10),
    ],
    dtype=[("name", "<U32"), ("obj", object), ("n_iter", int)],
)


# Evaluate results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment

loss_mc, t_run_mc = evaluate_algorithms_train(
    algorithms,
    problem_gen,
    n_gen,
    n_gen_learn,
    n_mc,
    solve=True,
    verbose=1,
    plotting=1,
    img_path="loss.png",
    rng=seed,
)
```
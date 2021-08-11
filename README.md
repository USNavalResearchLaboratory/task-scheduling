# Task Scheduling
This package provides a framework for implementing task scheduling algorithms and assessing their performance. It 
includes traditional schedulers as well as both supervised and reinforcement learning schedulers.


## Installation
The `task_scheduling` package has not been published to public code repositories. To install the local package, run
`pip install -e .` from the top-level repository directory.


## Tasks
Task objects must expose two attributes:
- `duration` - the time required to execute a task
- `t_release` - the earliest time at which a task may be executed

The tasks must implement a `__call__` method that provides a monotonic non-decreasing loss function quantifying the
penalty for delayed execution. 

One built-in task type is provided: `task_scheduling.tasks.ReluDrop`. It is so-named because it implements a loss 
function that increases linearly from zero according to a positive parameter `slope`, like the rectified linear 
function popularized by neural networks. After a "drop" time `t_drop`, a large constant loss `l_drop` is incurred. 
Example loss functions are shown below.

![Task loss functions](images/ex_relu_drop.png)


## Algorithms
The task scheduling problem is defined using two variables:
- `tasks` - an array of task objects
- `ch_avail` - an array of channel availability times

and the scheduling solution is defined using two arrays of length `len(tasks)`:
- `t_ex` - task execution times
- `ch_ex` - task execution channels

To be valid (as assessed using `util.results.check_schedule`), the execution times, execution channels, and task
durations must be such that no two tasks attempt execution at the same time, on the same channel.

Each algorithm is a Python `callable` implementing the same API; it takes two leading positional arguments `tasks` and
`ch_avail` and returns the schedule arrays `t_ex` and `ch_ex`. An example schedule is shown below.

![Task schedule](images/ex_schedule.png)

### Traditional schedulers
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

### Learning schedulers
Traditional schedulers typically suffer from one of two drawbacks: high computational load or poor performance. New
algorithms that learn from related problems may generalize well, finding near-optimal schedules in a 
shorter, more practical amount of runtime. 

The `learning.supervised` subpackage provides scheduler objects that combine policy networks (implemented with either 
[PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org/)) with state-action
environments following the API of [OpenAI Gym](https://gym.openai.com/). Scheduling environments are provided in
`learning.environments`, with the primary class `StepTasking`; this environment uses single task assignments as actions
and converts the scheduling problem (tasks and channel availabilities) into observed states, including the status of 
each task.


## Evaluation
The primary metrics used to evaluate a scheduling algorithm are its achieved loss and its runtime. The 
`util.results.evaluate_schedule` function calculates the total loss; the `util.generic.timing_wrapper` function allows 
any scheduler to be timed. 

While these functions can be invoked directly, the package provides a number of convenient functions in the 
`util.results` subpackage that automate this functionality, allow printing result tables to file, provide visuals, etc.
The functions `evaluate_algorithms_single` and `evaluate_algorithms_gen` assess for single scheduling problems and
across a set of generated problems, respectively. The function `evaluate_algorithms_train` adds an additional level of 
Monte Carlo iteration by re-training any learning schedulers a number of times. 

Example result outputs are shown below for `evaluate_algorithms_gen`. The first scatter plot shows raw loss-runtime 
pairs, one for each scheduling problem; the second plot shows excess loss values relative to optimal. The Markdown
format table provides the average values.

![Loss-runtime results](./images/ex_scatter.png)
![Loss-runtime results](./images/ex_scatter_relative.png)

```markdown
|                     |   Excess Loss (%) |    Loss |   Runtime |
|---------------------|-------------------|---------|-----------|
| BB Optimal          |             0.000 | 147.965 |     0.380 |
| Random              |             0.679 | 248.475 |     0.000 |
| ERT                 |             0.678 | 248.245 |     0.000 |
| MCTS: c=0.035, t=15 |             0.388 | 205.359 |     0.003 |
| Lit Policy          |             0.108 | 163.940 |     0.003 |
```

## Examples

### Basics
The following example shows a single scheduling problem and solution. Tasks are created using one of the provided
generators and a number of algorithms provide scheduling solutions. Built-in utilities help visualize the both the 
problem and the various solutions.

```python
from matplotlib import pyplot as plt

from task_scheduling import algorithms
from task_scheduling.tasks import summarize_tasks
from task_scheduling.generators import tasks as task_gens
from task_scheduling.util.plot import plot_task_losses, plot_schedule
from task_scheduling.util.results import check_schedule, evaluate_schedule

plt.style.use('seaborn')

seed = 12345


#%% Define scheduling problem
task_gen = task_gens.ContinuousUniformIID.relu_drop(rng=seed)

tasks = list(task_gen(8))
ch_avail = [0., 0.5]

summarize_tasks(tasks)
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
```

### Policy learning and Monte Carlo assessment
The following example demonstrates the definition of a scheduling problem generator, the creation of a learning
scheduler using PyTorch Lightning, and the comparison of traditional vs. learning schedulers using Monte Carlo
evaluation.

Note that the problem generator is used to instantiate the Environment, which is used to create and train the 
supervised learning policy. Also, note the structure of the `algorithms` array; each algorithm is has a name, a 
`callable`, and a number of iterations to perform per problem (averaging is best-practice for stochastic schedulers).

```python
from functools import partial
from itertools import product
# from operator import methodcaller

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from torch import nn, optim
from torch.nn import functional
import pytorch_lightning as pl

from task_scheduling.util.results import evaluate_algorithms_train
from task_scheduling.util.generic import RandomGeneratorMixin as RNGMix
from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.algorithms import mcts, random_sequencer, earliest_release, branch_bound_priority
from task_scheduling.learning import environments as envs
from task_scheduling.learning.supervised.torch import LitScheduler


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('seaborn')

seed = 12345


#%% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.discrete_relu_drop(n_tasks=8, n_ch=1, rng=seed)
problem_gen = problem_gens.Dataset.load('data/schedules/discrete_relu_c1t8', shuffle=True, repeat=True, rng=seed)


#%% Algorithms
env_params = {
    'features': None,  # defaults to task parameters
    'sort_func': 't_release',
    'time_shift': True,
    'masking': True,
    'action_type': 'valid',
    'seq_encoding': 'one-hot',
}

env = envs.StepTasking(problem_gen, **env_params)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(env.observation_space.shape).item(), 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, env.action_space.n),
            nn.LogSoftmax(dim=1),
        )
        self.loss_func = functional.nll_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


learn_params_pl = {'batch_size_train': 20,
                   'n_gen_val': 1/3,
                   'batch_size_val': 30,
                   'weight_func': None,
                   'max_epochs': 400,
                   'shuffle': True,
                   'callbacks': [pl.callbacks.EarlyStopping('val_loss', min_delta=0., patience=100)]
                   }


algorithms = np.array([
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False,
    #                                                                rng=RNGMix.make_rng(seed))), 1),
    ('Random', partial(random_sequencer, rng=RNGMix.make_rng(seed)), 10),
    ('ERT', earliest_release, 10),
    *((f'MCTS: c={c}, t={t}', partial(mcts, runtime=.002, c_explore=c, visit_threshold=t,
                                      rng=RNGMix.make_rng(seed)), 10) for c, t in product([.035], [15])),
    ('Lit Policy', LitScheduler(env, LitModel(), learn_params_pl), 10),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


#%% Evaluate results
n_gen_learn = 900  # the number of problems generated for learning, per iteration
n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment
l_ex_mc, t_run_mc = evaluate_algorithms_train(algorithms, n_gen_learn, problem_gen, n_gen=n_gen, n_mc=n_mc, solve=True,
                                              verbose=2, plotting=2, log_path=None)

```
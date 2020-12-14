## Offline work
- Supervised learning
  - Diagnose long scheduler runtime
- Reinforcement learning
  - Consider use of valid action hack
- Search for good loss-runtime results
  - Simplest generators first: Permuted -> Discrete -> Continuous


## Online work
- Create class for online scheduling problem
  - Attributes: full task list, channel availabilities
  - Methods: `serve` tasks
- Create class for schedulers
  - Interact with problem class via `__call__`?
  - Methods: `select` prioritizes tasks

**Serve tasks in-place, pass reference of problem object task attribute to the scheduler?**

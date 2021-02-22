# Notes
- Simple MLP
  - Masking effect negligible
  - Sorting effects
    - t_release: good
    - slope: moderate
    - duration: good
  - Shifting most potent
  - Sort, shift, and mask = BEST!
    - NOTE: effect of seq encoding is MINOR!
- TODO: investigate training sample weighting


---

Problem gen: Dataset
---
1 channel, 8 tasks

Channel: UniformIID(0.0, 0.0)

DiscreteIID
---
Task class: ReluDrop

|   duration |    Pr |
|------------|-------|
|      3.000 | 0.500 |
|      6.000 | 0.500 |

|   t_release |    Pr |
|-------------|-------|
|       0.000 | 0.500 |
|       4.000 | 0.500 |

|   slope |    Pr |
|---------|-------|
|   0.500 | 0.500 |
|   2.000 | 0.500 |

|   t_drop |    Pr |
|----------|-------|
|    6.000 | 0.500 |
|   12.000 | 0.500 |

|   l_drop |    Pr |
|----------|-------|
|   35.000 | 0.500 |
|   50.000 | 0.500 |

Number of problems: 1000

---

Non-learning results
---

- seed = 12345

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.674 | 244.300 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.113 | 166.180 |         0.020 |

![](../images/temp/2021-02-22_11-26-25.png)

---

# 2021-02-22_11-38-24

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.674 | 244.300 |         0.001 |
| ERT        |             0.658 | 243.560 |         0.001 |
| MCTS       |             0.113 | 166.180 |         0.049 |
| NN         |             0.088 | 162.545 |         0.017 |

![](../images/temp/2021-02-22_11-38-24.png)


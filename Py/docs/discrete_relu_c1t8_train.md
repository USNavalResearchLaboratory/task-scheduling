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

|            |   Relative Loss |    Loss |   Runtime |
|------------|-----------------|---------|-----------|
| BB Optimal |           0.000 | 148.531 |     0.359 |
| Random     |         101.375 | 249.906 |     0.000 |
| ERT        |          96.085 | 244.616 |     0.000 |
| MCTS       |          14.654 | 163.185 |     0.020 |

![](../images/discrete_relu_c1t8_non_learn.png)

---











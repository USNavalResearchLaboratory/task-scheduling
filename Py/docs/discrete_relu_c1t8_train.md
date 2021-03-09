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
- n_mc = 10
- n_gen = 100

|            |   Relative Loss |    Loss |   Runtime |
|------------|-----------------|---------|-----------|
| BB Optimal |           0.000 | 148.531 |     0.359 |
| Random     |         101.375 | 249.906 |     0.000 |
| ERT        |          96.085 | 244.616 |     0.000 |
| MCTS       |          15.383 | 163.914 |     0.016 |

![](../images/discrete_relu_c1t8_non_learn.png)

---









# 2021-03-09_13-19-59

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: True
- Masking: True
- Valid actions: False
- Sequence encoding: one-hot

Model
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 7, 50)             1350      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 6, 20)             2020      
_________________________________________________________________
flatten (Flatten)            (None, 120)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 968       
=================================================================
Total params: 4,338
Trainable params: 4,338
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Relative Loss |    Loss |   Runtime |
|------------|-----------------|---------|-----------|
| BB Optimal |           0.000 | 148.531 |     0.359 |
| NN         |           9.947 | 158.478 |     0.015 |


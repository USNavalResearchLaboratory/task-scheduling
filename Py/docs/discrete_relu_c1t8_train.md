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
| MCTS       |             0.113 | 166.180 |         0.021 |

![](../images/temp/2021-02-22_14-14-48.png)

---












# 2021-03-08_12-06-16

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: False
- Masking: True
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_2 (Conv1D)            (None, 7, 50)             1350      
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 6, 20)             2020      
_________________________________________________________________
flatten_1 (Flatten)          (None, 120)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 968       
=================================================================
Total params: 4,338
Trainable params: 4,338
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-03-08_12-06-16_train.png)

Results
---

|            |   Relative Loss |    Loss |   Runtime |
|------------|-----------------|---------|-----------|
| BB Optimal |           0.000 | 147.725 |     0.409 |
| Random     |          98.580 | 246.305 |     0.000 |
| ERT        |          87.325 | 235.050 |     0.000 |
| MCTS       |          14.375 | 162.100 |     0.021 |
| NN         |          13.200 | 160.925 |     0.016 |

![](../images/temp/2021-03-08_12-06-16.png)


# 2021-03-08_12-08-26

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: False
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

![](../images/temp/2021-03-08_12-08-26_train.png)

Results
---


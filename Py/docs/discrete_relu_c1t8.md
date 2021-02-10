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

Core results, seed=100
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| Random     |             0.684 | 245.975 |         0.000 |
| ERT        |             0.650 | 242.445 |         0.000 |
| MCTS       |             0.107 | 163.750 |         0.019 |

![](../images/temp/2021-02-10_15-32-55.png)

---

# 2021-02-10_15-43-09

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
- Task shifting: False
- Masking: False
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 104)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 30)                3150      
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-10_15-43-09_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.446 | 213.255 |         0.008 |

![](../images/temp/2021-02-10_15-43-09.png)


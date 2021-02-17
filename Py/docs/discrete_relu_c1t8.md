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

- seed = 100

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| Random     |             0.684 | 245.975 |         0.000 |
| ERT        |             0.650 | 242.445 |         0.000 |
| MCTS       |             0.107 | 163.750 |         0.019 |

![](../images/temp/2021-02-10_15-32-55.png)

---

# 2021-02-17_10-06-11

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
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-06-11_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.424 | 209.595 |         0.008 |

![](../images/temp/2021-02-17_10-06-11.png)

# 2021-02-17_10-08-49

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
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
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-08-49_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.403 | 207.460 |         0.008 |

![](../images/temp/2021-02-17_10-08-49.png)

# 2021-02-17_10-09-35

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
- Task shifting: True
- Masking: False
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-09-35_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.117 | 165.850 |         0.008 |

![](../images/temp/2021-02-17_10-09-35.png)

# 2021-02-17_10-11-01

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: False
- Masking: False
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-11-01_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.271 | 187.915 |         0.008 |

![](../images/temp/2021-02-17_10-11-01.png)

# 2021-02-17_10-13-15

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: slope
- Task shifting: False
- Masking: False
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-13-15_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.332 | 195.710 |         0.008 |

![](../images/temp/2021-02-17_10-13-15.png)

# 2021-02-17_10-15-09

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: duration
- Task shifting: False
- Masking: False
- Valid actions: False
- Sequence encoding: one-hot

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-15-09_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.285 | 191.075 |         0.008 |

![](../images/temp/2021-02-17_10-15-09.png)

# 2021-02-17_10-17-38

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
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-17-38_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.098 | 163.530 |         0.009 |

![](../images/temp/2021-02-17_10-17-38.png)

# 2021-02-17_10-20-49

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: True
- Masking: True
- Valid actions: False
- Sequence encoding: binary

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 48)                0         
_________________________________________________________________
dense (Dense)                (None, 30)                1470      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 1,718
Trainable params: 1,718
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-20-49_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.095 | 163.370 |         0.008 |

# 2021-02-17_10-35-36

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
- Task shifting: True
- Masking: True
- Valid actions: False
- Sequence encoding: None

Model 
---
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 40)                0         
_________________________________________________________________
dense (Dense)                (None, 30)                1230      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 1,478
Trainable params: 1,478
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-35-36_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| DNN        |             0.101 | 164.230 |         0.009 |

![](../images/temp/2021-02-17_10-35-36.png)

# 2021-02-17_10-43-19

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
flatten (Flatten)            (None, 104)               0         
_________________________________________________________________
dense (Dense)                (None, 30)                3150      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 248       
=================================================================
Total params: 3,398
Trainable params: 3,398
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_10-43-19_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| Random     |             0.684 | 245.975 |         0.000 |
| ERT        |             0.650 | 242.445 |         0.000 |
| MCTS       |             0.107 | 163.750 |         0.020 |
| DNN        |             0.086 | 162.005 |         0.009 |

![](../images/temp/2021-02-17_10-43-19.png)


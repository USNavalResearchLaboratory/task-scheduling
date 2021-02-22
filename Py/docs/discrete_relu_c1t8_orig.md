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










# 2021-02-17_15-28-00

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

![](../images/temp/2021-02-17_15-28-00_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| Random     |             0.704 | 249.407 |         0.000 |
| ERT        |             0.650 | 242.445 |         0.000 |
| MCTS       |             0.106 | 164.214 |         0.020 |
| DNN        |             0.093 | 162.930 |         0.009 |

![](../images/temp/2021-02-17_15-28-00.png)

# 2021-02-17_15-28-45

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

![](../images/temp/2021-02-17_15-28-45_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 149.385 |         0.337 |
| Random     |             0.704 | 249.407 |         0.000 |
| ERT        |             0.650 | 242.445 |         0.000 |
| MCTS       |             0.106 | 164.214 |         0.020 |
| DNN        |             0.096 | 163.220 |         0.009 |

![](../images/temp/2021-02-17_15-28-45.png)









# 2021-02-17_15-32-16

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

![](../images/temp/2021-02-17_15-32-16_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.704 | 249.058 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.105 | 164.594 |         0.020 |
| DNN        |             0.077 | 160.780 |         0.009 |

![](../images/temp/2021-02-17_15-32-16.png)

# 2021-02-17_15-33-49

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
dense (Dense)                (None, 100)               10500     
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 808       
=================================================================
Total params: 11,308
Trainable params: 11,308
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-33-49_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.095 | 163.660 |         0.009 |

![](../images/temp/2021-02-17_15-33-49.png)

# 2021-02-17_15-34-36

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
dense (Dense)                (None, 10)                1050      
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 88        
=================================================================
Total params: 1,138
Trainable params: 1,138
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-34-36_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.094 | 163.300 |         0.009 |

![](../images/temp/2021-02-17_15-34-36.png)

# 2021-02-17_15-37-31

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

![](../images/temp/2021-02-17_15-37-31_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.077 | 160.865 |         0.008 |

![](../images/temp/2021-02-17_15-37-31.png)

# 2021-02-17_15-38-13

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
dense_1 (Dense)              (None, 30)                930       
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 248       
=================================================================
Total params: 4,328
Trainable params: 4,328
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-38-13_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.083 | 162.300 |         0.011 |

![](../images/temp/2021-02-17_15-38-13.png)

# 2021-02-17_15-39-35

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
dense_1 (Dense)              (None, 10)                310       
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 88        
=================================================================
Total params: 3,548
Trainable params: 3,548
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-39-35_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.092 | 163.095 |         0.010 |

![](../images/temp/2021-02-17_15-39-35.png)

# 2021-02-17_15-46-44

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
conv1d (Conv1D)              (None, 7, 30)             810       
_________________________________________________________________
flatten (Flatten)            (None, 210)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 1688      
=================================================================
Total params: 2,498
Trainable params: 2,498
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-46-44_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.075 | 160.330 |         0.011 |

![](../images/temp/2021-02-17_15-46-44.png)

# 2021-02-17_15-48-42

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
conv1d (Conv1D)              (None, 5, 30)             1590      
_________________________________________________________________
flatten (Flatten)            (None, 150)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 1208      
=================================================================
Total params: 2,798
Trainable params: 2,798
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-48-42_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.067 | 159.330 |         0.011 |

![](../images/temp/2021-02-17_15-48-42.png)

# 2021-02-17_15-49-28

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
conv1d (Conv1D)              (None, 5, 100)            5300      
_________________________________________________________________
flatten (Flatten)            (None, 500)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 4008      
=================================================================
Total params: 9,308
Trainable params: 9,308
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-49-28_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.065 | 159.545 |         0.011 |

![](../images/temp/2021-02-17_15-49-28.png)

# 2021-02-17_15-50-20

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

![](../images/temp/2021-02-17_15-50-20_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.054 | 158.005 |         0.015 |

![](../images/temp/2021-02-17_15-50-20.png)

# 2021-02-17_15-51-20

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
dense (Dense)                (None, 6, 10)             210       
_________________________________________________________________
flatten (Flatten)            (None, 60)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 488       
=================================================================
Total params: 4,068
Trainable params: 4,068
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-51-20_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.049 | 157.050 |         0.019 |

![](../images/temp/2021-02-17_15-51-20.png)

# 2021-02-17_15-52-21

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
conv1d (Conv1D)              (None, 7, 80)             2160      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 6, 30)             4830      
_________________________________________________________________
dense (Dense)                (None, 6, 10)             310       
_________________________________________________________________
flatten (Flatten)            (None, 60)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 488       
=================================================================
Total params: 7,788
Trainable params: 7,788
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-52-21_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.062 | 159.330 |         0.018 |

![](../images/temp/2021-02-17_15-52-21.png)

# 2021-02-17_15-53-08

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
conv1d_2 (Conv1D)            (None, 5, 10)             410       
_________________________________________________________________
flatten (Flatten)            (None, 50)                0         
_________________________________________________________________
dense (Dense)                (None, 8)                 408       
=================================================================
Total params: 4,188
Trainable params: 4,188
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-53-08_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.064 | 159.380 |         0.018 |

![](../images/temp/2021-02-17_15-53-08.png)

# 2021-02-17_15-53-54

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

![](../images/temp/2021-02-17_15-53-54_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| DNN        |             0.064 | 159.260 |         0.015 |

![](../images/temp/2021-02-17_15-53-54.png)

# 2021-02-17_15-54-43

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

![](../images/temp/2021-02-17_15-54-43_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.704 | 249.058 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.105 | 164.594 |         0.020 |
| DNN        |             0.042 | 156.180 |         0.015 |

![](../images/temp/2021-02-17_15-54-43.png)

# 2021-02-17_15-55-55

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
dense (Dense)                (None, 7, 20)             1020      
_________________________________________________________________
flatten (Flatten)            (None, 140)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1128      
=================================================================
Total params: 3,498
Trainable params: 3,498
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-55-55_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.704 | 249.058 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.105 | 164.594 |         0.020 |
| DNN        |             0.051 | 157.740 |         0.015 |

![](../images/temp/2021-02-17_15-55-55.png)

# 2021-02-17_15-56-56

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
conv1d (Conv1D)              (None, 7, 50)             1350      
_________________________________________________________________
dense (Dense)                (None, 7, 20)             1020      
_________________________________________________________________
flatten (Flatten)            (None, 140)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1128      
=================================================================
Total params: 3,498
Trainable params: 3,498
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-56-56_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.704 | 249.058 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.105 | 164.594 |         0.020 |
| DNN        |             0.324 | 195.740 |         0.015 |

![](../images/temp/2021-02-17_15-56-56.png)

# 2021-02-17_15-57-49

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
dense (Dense)                (None, 7, 20)             1020      
_________________________________________________________________
flatten (Flatten)            (None, 140)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1128      
=================================================================
Total params: 3,498
Trainable params: 3,498
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-17_15-57-49_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| Random     |             0.704 | 249.058 |         0.000 |
| ERT        |             0.658 | 243.560 |         0.000 |
| MCTS       |             0.105 | 164.594 |         0.020 |
| DNN        |             0.051 | 157.650 |         0.015 |

![](../images/temp/2021-02-17_15-57-49.png)


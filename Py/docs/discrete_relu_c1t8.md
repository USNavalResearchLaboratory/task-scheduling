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

# 2021-02-22_14-36-35

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

![](../images/temp/2021-02-22_14-36-35_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| NN         |             0.435 | 210.780 |         0.008 |

# 2021-02-22_14-37-31

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

![](../images/temp/2021-02-22_14-37-31_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| NN         |             0.085 | 162.350 |         0.009 |


# 2021-02-22_14-38-49

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

![](../images/temp/2021-02-22_14-38-49_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| NN         |             0.066 | 159.620 |         0.015 |


# 2021-02-23_14-25-52

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

![](../images/temp/2021-02-23_14-25-52_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| NN         |             0.042 | 156.135 |         0.015 |

![](../images/temp/2021-02-23_14-25-52.png)


# 2021-02-24_09-58-54

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

![](../images/temp/2021-02-24_09-58-54_train.png)

Results
---

|            |   Excess Loss (%) |    Loss |   Runtime (s) |
|------------|-------------------|---------|---------------|
| BB Optimal |             0.000 | 150.115 |         0.374 |
| NN         |             0.063 | 159.295 |         0.016 |

![](../images/temp/2021-02-24_09-58-54.png)















# 2021-03-01_14-47-51

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

![](../images/temp/2021-03-01_14-47-51_train.png)

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 146.615 |     0.343 |
| Random     | 250.785 |     0.000 |
| ERT        | 243.510 |     0.000 |
| MCTS       | 161.240 |     0.020 |
| NN         | 159.480 |     0.009 |
![](../images/temp/2021-03-01_14-47-51.png)


# 2021-03-02_09-59-33

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 223.250 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 167.750 |     0.027 |
![](../images/temp/2021-03-02_09-59-33.png)


# 2021-03-02_10-01-23

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 249.600 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 160.250 |     0.029 |
![](../images/temp/2021-03-02_10-01-23.png)


# 2021-03-02_10-01-30

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 255.625 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 159.125 |     0.026 |
![](../images/temp/2021-03-02_10-01-30.png)


# 2021-03-02_10-01-56

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 223.250 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 167.750 |     0.026 |
![](../images/temp/2021-03-02_10-01-56.png)


# 2021-03-02_10-03-06

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 255.825 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 167.750 |     0.026 |
![](../images/temp/2021-03-02_10-03-06.png)


# 2021-03-02_10-03-43

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 223.250 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 167.750 |     0.028 |
![](../images/temp/2021-03-02_10-03-43.png)


# 2021-03-02_10-04-07

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 255.825 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 162.200 |     0.027 |
![](../images/temp/2021-03-02_10-04-07.png)


# 2021-03-02_10-07-28

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 255.825 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 162.200 |     0.026 |
![](../images/temp/2021-03-02_10-07-28.png)


# 2021-03-02_10-16-32

Results
---


|            |    Loss |   Runtime |
|------------|---------|-----------|
| BB Optimal | 138.750 |     0.212 |
| Random     | 255.825 |     0.000 |
| ERT        | 249.750 |     0.000 |
| MCTS       | 162.200 |     0.025 |
![](../images/temp/2021-03-02_10-16-32.png)


# 2021-03-02_13-21-46

Results
---


# 2021-03-02_13-22-38

Results
---


# 2021-03-02_13-23-08

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 248.616 |     0.000 |
| ERT    | 243.560 |     0.000 |
| MCTS   | 164.943 |     0.020 |
![](../images/temp/2021-03-02_13-23-08.png)


# 2021-03-02_13-26-37

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 248.616 |     0.000 |
| ERT    | 243.560 |     0.000 |
| MCTS   | 164.943 |     0.020 |
![](../images/temp/2021-03-02_13-26-37.png)


# 2021-03-02_13-28-46

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 252.055 |     0.001 |
| ERT    | 242.600 |     0.001 |
| MCTS   | 167.490 |     0.050 |

# 2021-03-02_13-33-15

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 252.055 |     0.000 |
| ERT    | 242.600 |     0.000 |
| MCTS   | 167.490 |     0.020 |
![](../images/temp/2021-03-02_13-33-15.png)


# 2021-03-02_13-56-14

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 252.055 |     0.000 |
| ERT    | 242.600 |     0.000 |
| MCTS   | 167.490 |     0.019 |
![](../images/temp/2021-03-02_13-56-14.png)


# 2021-03-02_14-01-36

Results
---

![](../images/temp/2021-03-02_14-01-36.png)


# 2021-03-02_14-01-49

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 242.130 |     0.000 |
| ERT    | 233.906 |     0.000 |
| MCTS   | 185.548 |     0.020 |
![](../images/temp/2021-03-02_14-01-49.png)


# 2021-03-02_14-02-06

Results
---

![](../images/temp/2021-03-02_14-02-06.png)


# 2021-03-02_14-02-25

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 242.130 |     0.000 |
| ERT    | 233.906 |     0.000 |
| MCTS   | 185.548 |     0.019 |
![](../images/temp/2021-03-02_14-02-25.png)


# 2021-03-02_14-02-39

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 242.130 |     0.000 |
| ERT    | 233.906 |     0.000 |
| MCTS   | 185.548 |     0.020 |
![](../images/temp/2021-03-02_14-02-39.png)


# 2021-03-02_14-04-09

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 242.130 |     0.000 |
| ERT    | 233.906 |     0.000 |
| MCTS   | 185.548 |     0.020 |
![](../images/temp/2021-03-02_14-04-09.png)


# 2021-03-02_14-05-06

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 242.130 |     0.000 |
| ERT    | 233.906 |     0.000 |
| MCTS   | 185.548 |     0.020 |
![](../images/temp/2021-03-02_14-05-06.png)


# 2021-03-02_14-05-21

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 238.525 |     0.000 |
| ERT    | 226.428 |     0.000 |
| MCTS   | 187.889 |     0.020 |
![](../images/temp/2021-03-02_14-05-21.png)


# 2021-03-02_14-05-34

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 238.525 |     0.000 |
| ERT    | 226.428 |     0.000 |
| MCTS   | 187.889 |     0.020 |
![](../images/temp/2021-03-02_14-05-34.png)


# 2021-03-02_14-34-34

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 238.525 |     0.000 |
| ERT    | 226.428 |     0.000 |
| MCTS   | 187.889 |     0.019 |
![](../images/temp/2021-03-02_14-34-34.png)


# 2021-03-02_14-34-54

Results
---


# 2021-03-02_14-39-38

Results
---


# 2021-03-02_14-39-47

Results
---


|        |    Loss |   Runtime |
|--------|---------|-----------|
| Random | 248.616 |     0.000 |
| ERT    | 243.560 |     0.000 |
| MCTS   | 164.943 |     0.021 |
![](../images/temp/2021-03-02_14-39-47.png)


# 2021-03-02_15-52-05

Results
---


# 2021-03-02_15-52-31

Results
---


# Notes
- Sorting effects
  - t_release: good
  - slope: moderate
  - duration: equally good!
- Shifting: very good
- Sort + shift = even better!
  - Negligible improvement from masking or encoding!
  
- Without sorting/shifting
  - Encoding-only and masking-only provide comparable improvement, minimal synergy
  
- TODO: investigate training sample weighting
  
- CNN: TODO

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

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.531 |     0.359 |
| Random             |             0.683 | 249.906 |     0.000 |
| ERT                |             0.647 | 244.616 |     0.000 |
| MCTS, c=0.05, t=15 |             0.086 | 161.307 |     0.022 |
| MCTS_v1, c=10      |             0.100 | 163.399 |     0.017 |
| MCTS               |             0.104 | 163.914 |     0.017 |

![](../images/discrete_relu_c1t8_non_learn.png)

---











# 2021-03-09_14-35-52

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.467 | 217.916 |     0.008 |


# 2021-03-09_14-53-04

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.261 | 187.263 |     0.008 |


# 2021-03-09_14-56-44

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.131 | 168.023 |     0.008 |



# 2021-03-09_15-06-08

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: t_release
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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.097 | 162.906 |     0.009 |


# 2021-03-09_15-10-05

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.093 | 162.387 |     0.009 |


# 2021-03-09_15-16-16

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.092 | 162.126 |     0.009 |


# 2021-03-09_15-25-00

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: duration
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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.088 | 161.664 |     0.009 |


# 2021-03-09_15-27-43

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: duration
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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.088 | 161.603 |     0.008 |















# 2021-03-10_08-49-56

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
- Task shifting: False
- Masking: False
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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.667 | 247.569 |     0.008 |


# 2021-03-10_08-56-14

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
- Task shifting: False
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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.420 | 210.918 |     0.008 |


# 2021-03-10_09-04-38

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

Training problems = 900

Results
---
- n_mc = 10
- n_gen = 100

|            |   Excess Loss (%) |    Loss |   Runtime |
|------------|-------------------|---------|-----------|
| BB Optimal |             0.000 | 148.531 |     0.359 |
| NN         |             0.405 | 208.737 |     0.008 |








---


# 2021-04-20_13-57-02

Results
---
n_gen = 100

|               |   Excess Loss (%) |    Loss |   Runtime |
|---------------|-------------------|---------|-----------|
| BB Optimal    |             0.000 | 148.470 |     0.336 |
| Random        |             0.672 | 248.256 |     0.000 |
| MCTS_v1, c=5  |             0.099 | 163.137 |     0.022 |
| MCTS_v1, c=10 |             0.088 | 161.543 |     0.022 |
| MCTS_v1, c=15 |             0.079 | 160.224 |     0.022 |


# 2021-04-20_14-03-08

Results
---
n_gen = 100

|               |   Excess Loss (%) |    Loss |   Runtime |
|---------------|-------------------|---------|-----------|
| BB Optimal    |             0.000 | 148.470 |     0.336 |
| Random        |             0.672 | 248.256 |     0.000 |
| MCTS_v1, c=5  |             0.099 | 163.137 |     0.023 |
| MCTS_v1, c=10 |             0.088 | 161.543 |     0.023 |
| MCTS_v1, c=15 |             0.079 | 160.224 |     0.023 |


# 2021-04-20_14-10-15

Results
---
n_gen = 100

|               |   Excess Loss (%) |    Loss |   Runtime |
|---------------|-------------------|---------|-----------|
| BB Optimal    |             0.000 | 148.470 |     0.336 |
| Random        |             0.672 | 248.256 |     0.000 |
| MCTS_v1, c=10 |             0.088 | 161.543 |     0.023 |
| MCTS_v1, c=20 |             0.080 | 160.345 |     0.023 |
| MCTS_v1, c=30 |             0.098 | 162.995 |     0.023 |


# 2021-04-20_14-13-47

Results
---

# 2021-04-20_14-13-52

Results
---

# 2021-04-20_14-13-58

Results
---
n_gen = 100

|                     |   Excess Loss (%) |    Loss |   Runtime |
|---------------------|-------------------|---------|-----------|
| BB Optimal          |             0.000 | 148.470 |     0.336 |
| Random              |             0.672 | 248.256 |     0.000 |
| MCTS, c=0, t=100    |             0.195 | 177.495 |     0.014 |
| MCTS, c=0.05, t=100 |             0.195 | 177.495 |     0.014 |
| MCTS, c=1, t=100    |             0.195 | 177.495 |     0.014 |


# 2021-04-20_14-17-05

Results
---
n_gen = 100

|                 |   Excess Loss (%) |    Loss |   Runtime |
|-----------------|-------------------|---------|-----------|
| BB Optimal      |             0.000 | 148.470 |     0.336 |
| Random          |             0.672 | 248.256 |     0.000 |
| MCTS, c=0, t=0  |             0.492 | 221.474 |     0.014 |
| MCTS, c=0, t=5  |             0.221 | 181.281 |     0.016 |
| MCTS, c=0, t=10 |             0.134 | 168.416 |     0.016 |
| MCTS, c=0, t=20 |             0.131 | 167.963 |     0.015 |


# 2021-04-21_12-10-57

Results
---
n_gen = 100

|                 |   Excess Loss (%) |    Loss |   Runtime |
|-----------------|-------------------|---------|-----------|
| BB Optimal      |             0.000 | 148.470 |     0.336 |
| Random          |             0.672 | 248.256 |     0.000 |
| MCTS, c=0, t=0  |             0.492 | 221.474 |     0.015 |
| MCTS, c=0, t=5  |             0.221 | 181.281 |     0.018 |
| MCTS, c=0, t=10 |             0.134 | 168.416 |     0.018 |
| MCTS, c=0, t=20 |             0.131 | 167.963 |     0.017 |


# 2021-04-21_12-18-13

Results
---
n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.470 |     0.336 |
| Random             |             0.672 | 248.256 |     0.000 |
| MCTS, c=0.05, t=15 |             0.124 | 166.916 |     0.016 |
| MCTS_v1, c=10      |             0.088 | 161.543 |     0.024 |


# 2021-04-21_12-19-36

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_12-19-36_train.png)

n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 145.965 |     0.385 |
| Random             |             0.703 | 248.537 |     0.000 |
| MCTS, c=0.05, t=15 |             0.125 | 164.193 |     0.016 |
| MCTS_v1, c=10      |             0.088 | 158.780 |     0.024 |
| NN Policy          |             0.068 | 155.820 |     0.020 |

![](../images/temp/2021-04-21_12-19-36.png)


# 2021-04-21_12-23-37

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_12-23-37_train.png)


# 2021-04-21_12-29-01

Results
---

# 2021-04-21_13-51-28

Results
---

# 2021-04-21_13-51-55

Results
---
n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.470 |     0.336 |
| Random             |             0.672 | 248.256 |     0.001 |
| MCTS, c=0.05, t=15 |             0.232 | 182.909 |     0.021 |
| MCTS_v1, c=10      |             0.215 | 180.367 |     0.021 |

![](../images/temp/2021-04-21_13-51-55.png)


# 2021-04-21_13-53-18

Results
---
n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.470 |     0.336 |
| Random             |             0.672 | 248.256 |     0.000 |
| MCTS, c=0.05, t=15 |             0.097 | 162.841 |     0.020 |
| MCTS_v1, c=10      |             0.095 | 162.516 |     0.021 |

![](../images/temp/2021-04-21_13-53-18.png)


# 2021-04-21_14-43-33

Results
---
n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.470 |     0.336 |
| Random             |             0.672 | 248.256 |     0.000 |
| MCTS, c=0.05, t=15 |             0.094 | 162.358 |     0.020 |
| MCTS_v1, c=10      |             0.097 | 162.819 |     0.020 |

![](../images/temp/2021-04-21_14-43-33.png)


# 2021-04-21_14-56-40

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_14-56-40_train.png)

n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 145.965 |     0.385 |
| Random             |             0.703 | 248.537 |     0.000 |
| MCTS, c=0.05, t=15 |             0.101 | 160.710 |     0.020 |
| MCTS_v1, c=10      |             0.099 | 160.459 |     0.020 |
| NN Policy          |             0.070 | 156.145 |     0.021 |

![](../images/temp/2021-04-21_14-56-40.png)


# 2021-04-21_14-59-47

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_14-59-47_train.png)

n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 145.965 |     0.385 |
| Random             |             0.703 | 248.537 |     0.000 |
| MCTS, c=0.05, t=15 |             0.096 | 159.906 |     0.020 |
| MCTS_v1, c=10      |             0.093 | 159.563 |     0.020 |
| NN Policy          |             0.063 | 155.115 |     0.020 |

![](../images/temp/2021-04-21_14-59-47.png)


# 2021-04-21_15-02-18

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_15-02-18_train.png)

n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 145.965 |     0.385 |
| Random             |             0.703 | 248.537 |     0.000 |
| MCTS, c=0.05, t=15 |             0.092 | 159.387 |     0.020 |
| MCTS_v1, c=10      |             0.095 | 159.843 |     0.020 |
| NN Policy          |             0.062 | 155.000 |     0.018 |

![](../images/temp/2021-04-21_15-02-18.png)


# 2021-04-21_15-09-39

Results
---
- n_mc = 10
- n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.709 |     0.354 |
| Random             |             0.673 | 248.818 |     0.000 |
| MCTS, c=0.05, t=15 |             0.089 | 161.953 |     0.020 |
| MCTS_v1, c=10      |             0.090 | 162.110 |     0.020 |

![](../images/temp/2021-04-21_15-09-39.png)


# 2021-04-21_15-18-06

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---

# 2021-04-21_15-20-10

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-04-21_15-20-10_train.png)

n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 145.965 |     0.385 |
| Random             |             0.703 | 248.537 |     0.000 |
| MCTS, c=0.05, t=15 |             0.098 | 160.246 |     0.020 |
| MCTS_v1, c=10      |             0.095 | 159.829 |     0.020 |
| NN Policy          |             0.069 | 155.980 |     0.019 |

![](../images/temp/2021-04-21_15-20-10.png)


# 2021-04-21_15-25-58

Results
---
n_gen = 100

|               |   Excess Loss (%) |    Loss |   Runtime |
|---------------|-------------------|---------|-----------|
| BB Optimal    |             0.000 | 148.470 |     0.336 |
| Random        |             0.672 | 248.256 |     0.000 |
| MCTS_v1, c=0  |             0.109 | 164.608 |     0.020 |
| MCTS_v1, c=10 |             0.090 | 161.897 |     0.020 |
| MCTS_v1, c=20 |             0.099 | 163.194 |     0.020 |
| MCTS_v1, c=30 |             0.110 | 164.841 |     0.021 |

![](../images/temp/2021-04-21_15-25-58.png)


# 2021-04-21_15-28-24

Results
---

# 2021-04-21_15-29-02

Results
---
- n_mc = 10
- n_gen = 100

|               |   Excess Loss (%) |    Loss |   Runtime |
|---------------|-------------------|---------|-----------|
| BB Optimal    |             0.000 | 148.709 |     0.354 |
| Random        |             0.673 | 248.818 |     0.000 |
| MCTS_v1, c=0  |             0.105 | 164.356 |     0.020 |
| MCTS_v1, c=10 |             0.088 | 161.834 |     0.020 |
| MCTS_v1, c=20 |             0.088 | 161.797 |     0.020 |
| MCTS_v1, c=30 |             0.105 | 164.256 |     0.020 |

![](../images/temp/2021-04-21_15-29-02.png)


# 2021-04-21_15-48-32

Results
---
- n_mc = 10
- n_gen = 100

|                   |   Excess Loss (%) |    Loss |   Runtime |
|-------------------|-------------------|---------|-----------|
| BB Optimal        |             0.000 | 148.709 |     0.354 |
| MCTS, c=0, t=10   |             0.126 | 167.477 |     0.020 |
| MCTS, c=0, t=15   |             0.100 | 163.649 |     0.020 |
| MCTS, c=0, t=20   |             0.096 | 162.923 |     0.020 |
| MCTS, c=0.2, t=10 |             0.117 | 166.104 |     0.020 |
| MCTS, c=0.2, t=15 |             0.120 | 166.626 |     0.020 |
| MCTS, c=0.2, t=20 |             0.125 | 167.302 |     0.020 |
| MCTS, c=0.4, t=10 |             0.143 | 169.959 |     0.020 |
| MCTS, c=0.4, t=15 |             0.145 | 170.245 |     0.020 |
| MCTS, c=0.4, t=20 |             0.145 | 170.338 |     0.020 |

![](../images/temp/2021-04-21_15-48-32.png)


# 2021-04-21_16-31-03

Results
---
- n_mc = 10
- n_gen = 100

|                   |   Excess Loss (%) |    Loss |   Runtime |
|-------------------|-------------------|---------|-----------|
| BB Optimal        |             0.000 | 148.709 |     0.354 |
| MCTS, c=0, t=10   |             0.124 | 167.205 |     0.020 |
| MCTS, c=0, t=15   |             0.097 | 163.065 |     0.020 |
| MCTS, c=0, t=20   |             0.095 | 162.847 |     0.020 |
| MCTS, c=0, t=25   |             0.104 | 164.105 |     0.020 |
| MCTS, c=0, t=30   |             0.111 | 165.165 |     0.020 |
| MCTS, c=0.1, t=10 |             0.095 | 162.882 |     0.020 |
| MCTS, c=0.1, t=15 |             0.097 | 163.120 |     0.020 |
| MCTS, c=0.1, t=20 |             0.102 | 163.882 |     0.020 |
| MCTS, c=0.1, t=25 |             0.111 | 165.278 |     0.020 |
| MCTS, c=0.1, t=30 |             0.120 | 166.497 |     0.020 |
| MCTS, c=0.2, t=10 |             0.116 | 165.912 |     0.020 |
| MCTS, c=0.2, t=15 |             0.118 | 166.271 |     0.020 |
| MCTS, c=0.2, t=20 |             0.123 | 166.948 |     0.020 |
| MCTS, c=0.2, t=25 |             0.128 | 167.700 |     0.020 |
| MCTS, c=0.2, t=30 |             0.131 | 168.233 |     0.020 |

![](../images/temp/2021-04-21_16-31-03.png)


# 2021-04-21_17-31-13

Results
---
- n_mc = 10
- n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.709 |     0.354 |
| MCTS, c=0.05, t=5  |             0.181 | 175.616 |     0.020 |
| MCTS, c=0.05, t=10 |             0.102 | 163.883 |     0.020 |
| MCTS, c=0.05, t=15 |             0.088 | 161.850 |     0.020 |
| MCTS, c=0.05, t=20 |             0.094 | 162.662 |     0.020 |
| MCTS, c=0.1, t=5   |             0.158 | 172.147 |     0.020 |
| MCTS, c=0.1, t=10  |             0.095 | 162.799 |     0.020 |
| MCTS, c=0.1, t=15  |             0.096 | 162.949 |     0.020 |
| MCTS, c=0.1, t=20  |             0.104 | 164.213 |     0.020 |

![](../images/temp/2021-04-21_17-31-13.png)


# 2021-04-21_18-22-09

Results
---
- n_mc = 10
- n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.709 |     0.354 |
| MCTS, c=0.03, t=10 |             0.108 | 164.712 |     0.020 |
| MCTS, c=0.03, t=15 |             0.090 | 162.035 |     0.020 |
| MCTS, c=0.03, t=20 |             0.091 | 162.263 |     0.020 |
| MCTS, c=0.06, t=10 |             0.097 | 163.123 |     0.020 |
| MCTS, c=0.06, t=15 |             0.088 | 161.728 |     0.020 |
| MCTS, c=0.06, t=20 |             0.095 | 162.886 |     0.020 |
| MCTS, c=0.09, t=10 |             0.097 | 163.065 |     0.020 |
| MCTS, c=0.09, t=15 |             0.093 | 162.541 |     0.020 |
| MCTS, c=0.09, t=20 |             0.101 | 163.736 |     0.020 |
| MCTS, c=0.12, t=10 |             0.098 | 163.271 |     0.020 |
| MCTS, c=0.12, t=15 |             0.100 | 163.605 |     0.020 |
| MCTS, c=0.12, t=20 |             0.106 | 164.465 |     0.020 |

![](../images/temp/2021-04-21_18-22-09.png)


# 2021-04-22_06-52-36

Results
---
- n_mc = 10
- n_gen = 100

|                    |   Excess Loss (%) |    Loss |   Runtime |
|--------------------|-------------------|---------|-----------|
| BB Optimal         |             0.000 | 148.478 |     0.370 |
| MCTS, c=0.03, t=10 |             0.107 | 164.411 |     0.020 |
| MCTS, c=0.03, t=15 |             0.087 | 161.454 |     0.020 |
| MCTS, c=0.03, t=20 |             0.092 | 162.068 |     0.020 |
| MCTS, c=0.06, t=10 |             0.097 | 162.943 |     0.020 |
| MCTS, c=0.06, t=15 |             0.088 | 161.614 |     0.020 |
| MCTS, c=0.06, t=20 |             0.095 | 162.559 |     0.020 |
| MCTS, c=0.09, t=10 |             0.093 | 162.341 |     0.020 |
| MCTS, c=0.09, t=15 |             0.093 | 162.227 |     0.020 |
| MCTS, c=0.09, t=20 |             0.101 | 163.409 |     0.020 |
| MCTS, c=0.12, t=10 |             0.097 | 162.895 |     0.020 |
| MCTS, c=0.12, t=15 |             0.099 | 163.230 |     0.020 |
| MCTS, c=0.12, t=20 |             0.105 | 164.132 |     0.020 |

![](../images/temp/2021-04-22_06-52-36.png)


# 2021-05-10_09-48-32

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-05-10_09-48-32_train.png)

n_gen = 100

|                     |   Excess Loss (%) |    Loss |   Runtime |
|---------------------|-------------------|---------|-----------|
| BB Optimal          |             0.000 | 145.965 |     0.385 |
| Random              |             0.703 | 248.537 |     0.000 |
| ERT                 |             0.651 | 240.930 |     0.000 |
| MCTS, c=0.035, t=15 |             0.102 | 160.921 |     0.020 |
| NN Policy           |             0.066 | 155.645 |     0.020 |

![](../images/temp/2021-05-10_09-48-32.png)


# 2021-05-20_10-14-33

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---
![](../images/temp/2021-05-20_10-14-33_train.png)


# 2021-05-20_10-21-50

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---

# 2021-05-20_10-22-53

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---

# 2021-05-20_10-23-38

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---

# 2021-05-20_10-24-34

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---

# 2021-05-20_10-26-17

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
conv1d_1 (Conv1D)            (None, 6, 20)             1220      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 5, 20)             820       
_________________________________________________________________
flatten (Flatten)            (None, 100)               0         
_________________________________________________________________
dense (Dense)                (None, 8)                 808       
=================================================================
Total params: 3,658
Trainable params: 3,658
Non-trainable params: 0
_________________________________________________________________
```

Training problems = 900

Results
---













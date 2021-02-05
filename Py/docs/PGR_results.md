# 2021-02-03_15-53-02

Problem gen: Random
---
1 channels, 4 tasks

Channel: UniformIID(0.0, 0.0)

ContinuousUniformIID
---
Task class: ReluDrop

|      |   duration |   t_release |   slope |   t_drop |   l_drop |
|------|------------|-------------|---------|----------|----------|
| low  |      3.000 |       0.000 |   0.500 |    6.000 |   35.000 |
| high |      6.000 |       4.000 |   2.000 |   12.000 |   50.000 |

Env: StepTasking
---

- Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
- Sorting: None
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
flatten (Flatten)            (None, 36)                0         
_________________________________________________________________
dense (Dense)                (None, 20)                740       
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 84        
=================================================================
Total params: 824
Trainable params: 824
Non-trainable params: 0
_________________________________________________________________
```

![](../images/temp/2021-02-03_15-53-02_train.png)

Results
---

|            |   Loss |   Runtime |
|------------|--------|-----------|
| BB Optimal | 40.625 |     0.004 |
| Random     | 73.536 |     0.000 |
| ERT        | 62.276 |     0.000 |
| MCTS       | 45.778 |     0.009 |
| DNN        | 53.679 |     0.004 |

![](../images/temp/2021-02-03_15-53-02.png)


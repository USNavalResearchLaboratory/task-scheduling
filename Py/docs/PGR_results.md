# CRM Results


Random
---
1 channels, 4 tasks

Channel: UniformIID(0.0, 0.0)


ContinuousUniformIID
---
Task class: `ReluDrop`

|      |   duration |   t_release |   slope |   t_drop |   l_drop |
|------|------------|-------------|---------|----------|----------|
| low  |      3.000 |       0.000 |   0.500 |    6.000 |   35.000 |
| high |      6.000 |       4.000 |   2.000 |   12.000 |   50.000 |


StepTasking
---

Features: ['duration', 't_release', 'slope', 't_drop', 'l_drop']
Sorting: None
Task shifting: True
Masking: True
Valid actions: False
Sequence encoding: one-hot

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

|            |   Loss |   Runtime |
|------------|--------|-----------|
| BB Optimal | 26.769 |     0.017 |
| Random     | 65.454 |     0.000 |
| ERT        | 47.159 |     0.000 |
| MCTS       | 29.714 |     0.032 |
| DNN Policy | 51.520 |     0.008 |

![](../images/temp/dat_result.png)

## Introduction

[ALGORITHM]

### Train Dataset

|  trainset  | instance_num | repeat_num | note  |
| :--------: | :----------: | :--------: | :---: |
| icdar_2011 |     3567     |     20     | real  |
| icdar_2013 |     848      |     20     | real  |
| icdar2015  |     4468     |     20     | real  |
| coco_text  |    42142     |     20     | real  |
|   IIIT5K   |     2000     |     20     | real  |
| SynthText  |   2400000    |     1      | synth |

### Test Dataset

| testset | instance_num |            note             |
| :-----: | :----------: | :-------------------------: |
| IIIT5K  |     3000     |           regular           |
|   SVT   |     647      |           regular           |
|  IC13   |     1015     |           regular           |
|  IC15   |     2077     |          irregular          |
|  SVTP   |     645      | irregular, 639 in [[1]](#1) |
|  CT80   |     288      |          irregular          |

## Results and models

|   methods   |        | Regular Text |      |     |      | Irregular Text |      |       download       |
| :---------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :------------------: |
|             | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| Transformer |  93.3  |     85.8     | 91.3 |     | 73.2 |      76.6      | 87.8 | [model]() \| [log]() |

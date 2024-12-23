# RTTDet

# Description

# Environments

```
shapely
```


# Process

## 1. Dataset

- [dataset](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/weaklysupervised/data/utils/basedataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/weaklysupervised/models/rttdet/model/rttdet.py)

<p align='center'>
    <img width='1500' src='assets/rttdet.png'>
</p>

# Run

## Synth Model Train 

```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdet --yaml_config configs/textdetection/weaklysupervised/rttdet/synth.yaml
```

## Synth-Icdar13 Model Train 

```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdet --yaml_config configs/textdetection/weaklysupervised/rttdet/icdar13.yaml
```

## Synth-Icdar15 Model Train 

```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdet --yaml_config configs/textdetection/weaklysupervised/rttdet/icdar15.yaml
```

## Synth-Icdar17 Model Train 

```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdet --yaml_config configs/textdetection/weaklysupervised/rttdet/icdar17.yaml
```

## Demo

![](assets/1.png)
![](assets/2.png)
![](assets/3.png)
![](assets/4.png)

# Results

TBD

|    | target           |   Hmean       |        Recall |     Precision |
|---:|:-----------------|--------------:|--------------:|--------------:|
|  0 | synth            |         100   |         98.70 |         96.02 |
|  1 | synth-icdar15    |         94.80 |         98.20 |         94.10 |

# YOLOV3

# Description

# Environments

```
einops
kornia
torchmetrics==0.10.3
timm
```


# Process

## 1. Dataset

- [yolov3dataset](https://github.com/pntrungbk15/TNVision/blob/main/tasks/objectdetection/supervised/data/dataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/tasks/objectdetection/supervised/models/yolov3/model/yolov3.py)

<p align='center'>
    <img width='1500' src='assets/yolov3.png'>
</p>

# Run

```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov3 --yaml_config configs/objectdetection/supervised/yolov3/pascal.yaml
```

## Demo

### zipper
<p align="left">
  <img src=assets/zipper.gif width="100%" />
</p>

### wood
<p align="left">
  <img src=assets/wood.gif width="100%" />
</p>

# Results

TBD

|    | target     |   AUROC-image |   AUROC-pixel |   AUPRO-pixel |
|---:|:-----------|--------------:|--------------:|--------------:|
|  0 | bottle     |         100   |         98.36 |         94.58 |
|  1 | cable      |         99.08 |         98.70 |         93.74 |
|  2 | capsule    |         93.90 |         98.66 |         91.89 |
|  3 | carpet     |         97.27 |         98.53 |         91.97 |
|  4 | grid       |         98.16 |         98.19 |         92.99 |
|  5 | hazelnut   |         100   |         98.81 |         95.16 |
|  6 | leather    |         99.97 |         98.88 |         94.25 |
|  7 | meatal_nut |         99.61 |         98.61 |         91.52 |
|  8 | pill       |         93.29 |         98.10 |         96.25 |
|  9 | screw      |         82.84 |         98.29 |         92.78 |
| 10 | tile       |         99.86 |         95.25 |         85.41 |
| 11 | toothbrush |         93.89 |         98.85 |         89.40 |
| 12 | transitor  |         99.29 |         97.85 |         94.70 |
| 13 | wood       |         98.42 |         94.66 |         87.42 |
| 14 | zipper     |         98    |         98.54 |         93.67 |
|    | **Average**    |         96.91 |         98.02 |         92.38 |

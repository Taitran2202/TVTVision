# PatchCore
Unofficial Re-implementation for [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265.pdf)

## Description

Có khả năng nhận diện các bộ phận bị lỗi là một thành phần quan trọng trong quy trình sản xuất công nghiệp quy mô lớn. Một thách thức đặc biệt mà chúng tôi giải quyết trong công việc này là vấn đề khởi động lạnh: tạo mô hình bằng cách sử dụng chỉ các hình ảnh mẫu không có lỗi. Trong khi các giải pháp được tạo bằng tay cho từng lớp riêng lẻ là có thể, mục tiêu của chúng tôi là xây dựng các hệ thống hoạt động tốt đồng thời trên nhiều nhiệm vụ khác nhau một cách tự động. Các phương pháp tốt nhất kết hợp nhúng từ các mô hình ImageNet với mô hình phát hiện ngoại lệ. Trong bài báo này, chúng tôi mở rộng trên dòng công việc này và đề xuất PatchCore, sử dụng một ngân hàng bộ nhớ biểu diễn tối đa của các tính năng miếng mẫu không lỗi. PatchCore cung cấp thời gian suy luận cạnh tranh trong khi đạt được hiệu suất tiên tiến nhất cho cả phát hiện và xác định vị trí. Trên thử thách, MVTec AD benchmark được sử dụng rộng rãi, PatchCore đạt được điểm AUROC cho phát hiện tỷ lệ bất thường trên cấp độ hình ảnh lên đến 99,6%, giảm lỗi gần một nửa so với đối thủ tốt nhất tiếp theo.

# Environments

```
einops
kornia
torchmetrics==0.10.3
timm
```


# Process

## 1. Dataset

- [mvtecdataset](https://github.com/pntrungbk15/TNVision/blob/main/task/anomaly/unsupervised/data/dataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVisionV2/blob/main/task/anomaly/unsupervised/models/patchcorers/model/patchcorers.py)

<p align='center'>
    <img width='1500' src='assets/patchcorers.png'>
</p>


# Run

```bash
python main.py --task_type anomaly --model_type unsupervised --model_name patchcorers --yaml_config configs/anomaly/unsupervised/patchcorers/bottle.yaml
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

### transistor
<p align="left">
  <img src=assets/transistor.gif width="100%" />
</p>

### toothbrush
<p align="left">
  <img src=assets/toothbrush.gif width="100%" />
</p>

### tile
<p align="left">
  <img src=assets/tile.gif width="100%" />
</p>

### screw
<p align="left">
  <img src=assets/screw.gif width="100%" />
</p>

### pill
<p align="left">
  <img src=assets/pill.gif width="100%" />
</p>

### metal_nut
<p align="left">
  <img src=assets/metal_nut.gif width="100%" />
</p>

### leather
<p align="left">
  <img src=assets/leather.gif width="100%" />
</p>

### hazelnut
<p align="left">
  <img src=assets/hazelnut.gif width="100%" />
</p>

### grid
<p align="left">
  <img src=assets/grid.gif width="100%" />
</p>

### carpet
<p align="left">
  <img src=assets/carpet.gif width="100%" />
</p>

### capsule
<p align="left">
  <img src=assets/capsule.gif width="100%" />
</p>

### cable
<p align="left">
  <img src=assets/cable.gif width="100%" />
</p>

### bottle
<p align="left">
  <img src=assets/bottle.gif width="100%" />
</p>

# Results

### Image-Level AUC

|                          |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ------------------------ | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
|  | 0.000 | 0.000  | 0.000 |  0.000  | 0.000 | 0.000 | 0.000  | 0.000 |  0.000  |  0.000   |   0.000   | 0.000 | 0.000 |   0.000    |   0.000    | 0.000  |

### Pixel-Level AUC

|                          |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ------------------------ | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
|  | 0.000 | 0.000  | 0.000 |  0.000  | 0.000 | 0.000 | 0.000  | 0.000 |  0.000  |  0.000   |   0.000   | 0.000 | 0.000 |   0.000    |   0.000    | 0.000  |

### Pixel F1 Score

|                          |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ------------------------ | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
|  | 0.000 | 0.000  | 0.000 |  0.000  | 0.000 | 0.000 | 0.000  | 0.000 |  0.000  |  0.000   |   0.000   | 0.000 | 0.000 |   0.000    |   0.000    | 0.000  |
# SAPatchCore
Unofficial Re-implementation for [SA-PatchCore: Anomaly Detection in Dataset With Co-Occurrence Relationships Using Self-Attention](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10007829)

## Description

Các phương pháp phát hiện bất thường không giám sát sử dụng học sâu đã được đề xuất gần đây, và độ chính xác của kỹ thuật phát hiện bất thường cho các bất thường cục bộ đã được cải thiện. Tuy nhiên, không có tập dữ liệu phát hiện bất thường nào bao gồm các bất thường liên quan đến sự kết hợp. Do đó, độ chính xác của việc phát hiện bất thường cho các bất thường liên quan đến sự kết hợp vẫn chưa tiến bộ.
Vì vậy, chúng tôi đề xuất SA-PatchCore, giới thiệu tự chú ý vào mô hình phát hiện bất thường cục bộ tiên tiến nhất là PatchCore. Nó phát hiện bất thường trong mối quan hệ cùng xuất hiện và bất thường trong các khu vực cục bộ với lợi ích của mô-đun tự chú ý, có thể xem xét ngữ cảnh giữa các từ tách rời, được giới thiệu đầu tiên trong lĩnh vực xử lý ngôn ngữ tự nhiên. Vì không có tập dữ liệu phát hiện bất thường nào bao gồm bất thường trong mối quan hệ cùng xuất hiện, chúng tôi chuẩn bị một tập dữ liệu mới gọi là Co-occurrence Anomaly Detection Screw Dataset (CAD-SD). Hơn nữa, chúng tôi đã tiến hành thử nghiệm phát hiện bất thường sử dụng tập dữ liệu mới.
SA-PatchCore đạt hiệu suất phát hiện bất thường cao so với PatchCore trong CAD-SD. Hơn nữa, mô hình mà chúng tôi đề xuất hiển thị gần như cùng hiệu suất phát hiện bất thường với PatchCore trong tập dữ liệu MVTec Anomaly Detection, bao gồm các bất thường trong một khu vực cục bộ

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

- [model](https://github.com/pntrungbk15/TNVisionV2/blob/main/task/anomaly/unsupervised/models/sapatchcore/model/sapatchcore.py)

<p align='center'>
    <img width='1500' src='assets/sapatchcore.png'>
</p>


# Run

```bash
python main.py --task_type anomaly --model_type unsupervised --model_name sapatchcore --yaml_config configs/anomaly/unsupervised/sapatchcore/bottle.yaml
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
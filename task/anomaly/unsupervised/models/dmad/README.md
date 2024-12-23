# DMAD
Unofficial Re-implementation for [Diversity-Measurable Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Diversity-Measurable_Anomaly_Detection_CVPR_2023_paper.pdf)

# Description

Các mô hình phát hiện dấu vết dựa trên tái tạo đạt được mục tiêu của họ bằng cách ức chế khả năng tổng quát hóa đối với các biểu mẫu bất thường. Tuy nhiên, các biểu mẫu bình thường đa dạng do đó không được tái tạo tốt. Mặc dù một số nỗ lực đã được thực hiện để giảm bớt vấn đề này bằng cách mô hình hóa đa dạng mẫu, nhưng họ gặp khó khăn với việc học ngắn hạn do truyền thông không mong muốn của thông tin không bình thường. Trong bài báo này, để xử lý vấn đề cân đối tốt hơn, chúng tôi đề xuất một khung làm việc gọi là Diversity-Measurable Anomaly Detection (DMAD) để tăng cường đa dạng trong việc tái tạo trong khi tránh việc tổng quát hóa không mong muốn đối với các sự cố. Để làm được điều này, chúng tôi thiết kế Mô-đun Biến đổi Pyramid (PDM), mô hình hóa các biểu mẫu bình thường đa dạng và đo độ nghiêm trọng của sự cố bằng cách ước tính các lĩnh vực biến đổi đa quy mô từ biểu mẫu tham chiếu đã tái tạo đến đầu vào gốc. Kết hợp với một mô-đun nén thông tin, PDM về cơ bản tách biến đổi khỏi việc nhúng mẫu mẫu và làm cho điểm bất thường cuối cùng trở nên đáng tin cậy hơn. Kết quả thực nghiệm trên cả video giám sát và hình ảnh công nghiệp chứng minh hiệu quả của phương pháp của chúng tôi. Ngoài ra, DMAD hoạt động cũng tốt trước dữ liệu bị nhiễu và các mẫu bình thường giống như dấu vết.

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

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/anomaly/unsupervised/models/dmad/model/dmad.py)

<p align='center'>
    <img width='1500' src='assets/dmad.png'>
</p>

# Run

```bash
python main.py --task_type anomaly --model_type unsupervised --model_name dmad --yaml_config configs/anomaly/unsupervised/dmad/bottle.yaml
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
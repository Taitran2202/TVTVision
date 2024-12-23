# CATS
Unofficial Re-implementation for [Unmixing Convolutional Features for Crisp Edge Detection](https://arxiv.org/pdf/2011.09808.pdf)

# Description

Bài báo này giới thiệu một chiến lược theo dõi dựa trên ngữ cảnh (Context-Aware Tracing Strategy - CATS) cho việc phát hiện cạnh rõ nét bằng các bộ phát hiện cạnh sâu, dựa trên quan sát rằng độ mơ hồ của các bộ phát hiện cạnh sâu chủ yếu do hiện tượng trộn lẫn của các mạng nơ-ron tích chập: hiện tượng trộn lẫn trong phân loại cạnh và trộn lẫn bên trong quá trình hợp nhất các dự đoán từ các bên. CATS bao gồm hai module: một hàm mất mát theo dõi mới mà thực hiện giải trộn tính năng bằng cách theo dõi biên giới để tăng cường quá trình học cạnh bên, và một khối hợp nhất phù hợp với ngữ cảnh giúp giải quyết hiện tượng trộn lẫn bên trong quá trình tổng hợp độc lập của cạnh từ các bên đã học được. Các thực nghiệm chứng minh rằng CATS đề xuất có thể tích hợp vào các bộ phát hiện cạnh sâu hiện đại để cải thiện độ chính xác trong việc xác định vị trí. Với kiến trúc cơ bản VGG16, trên tập dữ liệu BSDS500, CATS của chúng tôi cải thiện chỉ số F-measure (ODS) của các bộ phát hiện cạnh sâu RCF và BDCN lần lượt là 12% và 6% so với việc đánh giá không sử dụng giải thuật hạn chế cực đại không đồng dạng cực tiểu theo hình thái học cho việc phát hiện cạnh.


# Environments

```
```


# Process

## 1. Dataset

- [edgedataset](https://github.com/pntrungbk15/TNVision/blob/main/task/edgedetection/supervised/data/dataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/edgedetection/supervised/models/cats/model/cats.py)

<p align='center'>
    <img width='1500' src='assets/cats.png'>
</p>

# Run

```bash
python main.py --task_type edgedetection --model_type supervised --model_name cats --yaml_config configs/edgedetection/supervised/cats/bsds.yaml
```

## Demo

### BDS500
<p align="left">
  <img src=assets/bds.gif width="100%" />
</p>

# Results

TBD

|    | target     |   AUROC-image |   AUROC-pixel |   AUPRO-pixel |
|---:|:-----------|--------------:|--------------:|--------------:|
|  0 | bottle     |         100   |         98.70 |         96.02 |
|  1 | capsule    |         94.80 |         98.20 |         94.10 |
|  2 | wood       |         99.82 |         97.12 |         93.41 |
|  3 | pill       |         97.25 |         98.21 |         95.30 |
|  4 | leather    |         100   |         99.31 |         98.83 |
|  5 | hazelnut   |         98.79 |         97.31 |         96.67 |
|    | **Average**    |         98.44 |         98.14 |         95.72 |

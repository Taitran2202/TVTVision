# DarkNet19

# Description

Phân loại hình ảnh là một nhiệm vụ cơ bản nhằm cố gắng hiểu toàn bộ hình ảnh như một tổng thể. Mục tiêu là phân loại hình ảnh bằng cách gán nó cho một nhãn cụ thể. Thông thường, Phân loại hình ảnh đề cập đến các hình ảnh trong đó chỉ có một đối tượng xuất hiện và được phân tích. Ngược lại, phát hiện đối tượng liên quan đến cả nhiệm vụ phân loại và bản địa hóa và được sử dụng để phân tích các trường hợp thực tế hơn trong đó có thể tồn tại nhiều đối tượng trong một hình ảnh.

# Environments

```
```

# Process
## 1. Dataset

- [Classify dataset](https://github.com/pntrungbk15/TNVision/blob/main/tasks/classify/supervised/data/dataset.py)

## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/tasks/classify/supervised/models/darknet19/model/darknet19.py)

# Run

```bash
python main.py --task_type classify --model_type supervised --model_name darknet19 --yaml_config configs/classify/supervised/darknet19/garbage.yaml
```

## Demo

### Garbage
<p align="left">
  <img src=assets/garbage.gif width="100%" />
</p>


# Results

TBD

|    | target     |   ACCURACY 
|---:|:-----------|--------------
|  0 | garbage    |         86.17  

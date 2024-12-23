# PSENET
Unofficial Re-implementation for [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/pdf/1903.12473.pdf)

# Description

Phát hiện văn bản trong cảnh đã chứng kiến tiến bộ nhanh chóng, đặc biệt là với sự phát triển gần đây của các mạng neural tích chập. Tuy nhiên, vẫn tồn tại hai thách thức ngăn cản thuật toán từ việc ứng dụng trong các lĩnh vực công nghiệp. Một mặt, hầu hết các thuật toán hiện đại yêu cầu hộp giới hạn hình bốn cạnh, điều này không chính xác để xác định vị trí của văn bản có hình dạng tùy ý. Mặt khác, hai trường hợp văn bản gần nhau có thể dẫn đến phát hiện sai lầm, che phủ cả hai trường hợp. Truyền thống, phương pháp dựa trên phân đoạn có thể giải quyết vấn đề đầu tiên nhưng thường không thể giải quyết thách thức thứ hai. Để giải quyết hai thách thức này, trong bài báo này, chúng tôi đề xuất một Mạng Mở Rộng Quy Mô Tiến Tiến (PSENet) mới, có khả năng phát hiện chính xác các trường hợp văn bản với hình dạng tùy ý. Cụ thể hơn, PSENet tạo ra các hạt nhân với các quy mô khác nhau cho mỗi trường hợp văn bản, và dần dần mở rộng hạt nhân quy mô nhỏ nhất đến trường hợp văn bản với hình dạng hoàn chỉnh. Do sự thật rằng có các biên độ hình học lớn giữa các hạt nhân quy mô nhỏ nhất, phương pháp của chúng tôi hiệu quả trong việc phân tách các trường hợp văn bản gần nhau, làm cho việc sử dụng phương pháp dựa trên phân đoạn để phát hiện các trường hợp văn bản có hình dạng tùy ý dễ dàng hơn. Thực nghiệm đa dạng trên CTW1500, Total-Text, ICDAR 2015 và ICDAR 2017 MLT đã xác nhận hiệu quả của PSENet.

# Environments

```
shapely
```


# Process

## 1. Dataset

- [psenetdataset](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/supervised/models/psenet/data/module/basedataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/supervised/models/psenet/model/psenet.py)

<p align='center'>
    <img width='1500' src='assets/psenet.png'>
</p>

# Run

## Synth Model Train 

```bash
python main.py --task_type textdetection --model_type supervised --model_name psenet --yaml_config configs/textdetection/supervised/psenet/synth.yaml
```

## Synth-Icdar15 Model Train 

```bash
python main.py --task_type textdetection --model_type supervised --model_name psenet --yaml_config configs/textdetection/supervised/psenet/synth_icdar15.yaml
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

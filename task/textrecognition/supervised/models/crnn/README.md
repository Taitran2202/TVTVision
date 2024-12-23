# CRNN + CTC
Unofficial Re-implementation for [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)

# Description

Nhận dạng chuỗi dựa trên hình ảnh đã là một chủ đề nghiên cứu lâu dài trong lĩnh vực thị giác máy tính. Trong bài báo này, chúng tôi điều tra vấn đề nhận dạng văn bản cảnh quan, đó là một trong những nhiệm vụ quan trọng và thách thức nhất trong nhận dạng chuỗi dựa trên hình ảnh. Chúng tôi đề xuất một kiến trúc mạng thần kinh mới, kết hợp trích xuất đặc trưng, mô hình chuỗi và chuyển văn bản thành một khung làm việc thống nhất. So với các hệ thống trước đây cho nhận dạng văn bản cảnh quan, kiến trúc đề xuất có bốn tính năng đáng chú ý: (1) Nó có thể huấn luyện từ đầu đến cuối, khác với hầu hết các thuật toán hiện có mà các thành phần của chúng được huấn luyện và điều chỉnh một cách riêng lẻ. (2) Nó tự nhiên xử lý các chuỗi có độ dài tùy ý, không liên quan đến việc phân đoạn ký tự hoặc chuẩn hóa tỷ lệ ngang. (3) Nó không bị giới hạn bởi bất kỳ từ điển được xác định trước nào và đạt được hiệu suất đáng kể cả trong nhiệm vụ nhận dạng văn bản cảnh quan không có từ điển và có từ điển. (4) Nó tạo ra một mô hình hiệu quả nhưng nhỏ hơn nhiều, phù hợp hơn cho các kịch bản ứng dụng thực tế. Các thí nghiệm trên các bộ kiểm tra tiêu chuẩn, bao gồm IIIT-5K, Street View Text và ICDAR, chứng minh sự ưu việt của thuật toán đề xuất so với các công trình trước đây. Hơn nữa, thuật toán đề xuất thực hiện tốt trong nhiệm vụ nhận dạng nhạc bản dựa trên hình ảnh, điều này rõ ràng xác nhận tính tổng quát của nó.

# Environments

```
torch==1.16.0
```


# Process

## 1. Dataset

- [CRNN dataset](https://github.com/pntrungbk15/TNVision/blob/main/task/textrecognition/crnn/dataset/module/crnnbasedataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/textrecognition/crnn/model/crnn.py)

<p align='center'>
    <img width='1500' src='assets/crnn.png'>
</p>

# Run

## Synth Model Train 

```bash
python main.py --task_type textrecognition --model_type supervised --model_name crnn --yaml_config configs/textrecognition/supervised/crnn/synth90k.yaml
```

## Synth-Icdar13 Model Train 

```bash
python main.py --task_type textrecognition --model_type supervised --model_name crnn --yaml_config configs/textrecognition/supervised/crnn/icdar13.yaml
```

## Synth-Icdar15 Model Train 

```bash
python main.py --task_type textrecognition --model_type supervised --model_name crnn --yaml_config configs/textrecognition/supervised/crnn/icdar15.yaml
```

## Demo

### synth90k
<p align="left">
  <img src=assets/synth90k.gif width="100%" />
</p>

### icdar13
<p align="left">
  <img src=assets/icdar13.gif width="100%" />
</p>


# Results

TBD

|    | target     |   ACCURACY 
|---:|:-----------|--------------
|  0 | synth      |         93.1   
|  1 | icdar13    |         92.0 


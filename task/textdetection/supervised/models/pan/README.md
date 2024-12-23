# PAN
Unofficial Re-implementation for [Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/pdf/1908.05900.pdf)

# Description

Phát hiện văn bản trong cảnh, một bước quan trọng của các hệ thống đọc văn bản trong cảnh, đã chứng kiến sự phát triển nhanh chóng với các mạng neural tích chập. Tuy nhiên, vẫn tồn tại hai thách thức chính và làm trở ngại cho việc triển khai nó vào các ứng dụng thực tế. Vấn đề đầu tiên là sự cân đối giữa tốc độ và độ chính xác. Vấn đề thứ hai là việc mô hình hóa các trường hợp văn bản có hình dáng tùy ý. Gần đây, một số phương pháp đã được đề xuất để giải quyết việc phát hiện văn bản có hình dáng tùy ý, nhưng chúng hiếm khi xem xét tốc độ của toàn bộ quy trình, điều này có thể không đủ trong các ứng dụng thực tế. Trong bài báo này, chúng tôi đề xuất một bộ phát hiện văn bản có hình dáng tùy ý hiệu quả và chính xác, được gọi là Mạng Tích hợp Pixel (PAN), được trang bị một đầu phân đoạn có chi phí tính toán thấp và một phần xử lý sau có thể học được. Cụ thể hơn, đầu phân đoạn bao gồm Mô-đun Tăng cường Pyramide Đặc trưng (FPEM) và Mô-đun Kết hợp Đặc trưng (FFM). FPEM là một mô-đun dạng U có thể được xếp chồng lên, có thể giới thiệu thông tin đa cấp để hướng dẫn phân đoạn tốt hơn. FFM có thể thu thập các đặc trưng được cung cấp bởi FPEM ở các độ sâu khác nhau thành một đặc trưng cuối cùng cho phân đoạn. Phần xử lý sau có thể học được được thực hiện bằng Cụm hợp pixel (PA), có thể chính xác tập trung các pixel văn bản bằng các vectơ tương đồng được dự đoán. Thực nghiệm trên một số tiêu chuẩn thông thường xác nhận sự ưu việt của PAN được đề xuất. Đáng chú ý rằng phương pháp của chúng tôi có thể đạt được một giá trị F-measure cạnh tranh là 79.9% tại 84.2 FPS trên CTW1500.

# Environments

```
shapely
```


# Process

## 1. Dataset

- [pandataset](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/supervised/models/pan/data/module/basedataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/textdetection/supervised/models/pan/model/pan.py)

<p align='center'>
    <img width='1500' src='assets/pan.png'>
</p>

# Run

## Synth Model Train 

```bash
python main.py --task_type textdetection --model_type supervised --model_name pan --yaml_config configs/textdetection/supervised/pan/synth.yaml
```

## Synth-Icdar15 Model Train 

```bash
python main.py --task_type textdetection --model_type supervised --model_name pan --yaml_config configs/textdetection/supervised/pan/synth_icdar15.yaml
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

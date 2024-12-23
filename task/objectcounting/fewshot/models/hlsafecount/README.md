# HLSAFECount
Unofficial Re-implementation for [Few-shot Object Counting with Similarity-Aware Feature Enhancement](https://arxiv.org/pdf/2201.08959v5.pdf)

# Description

Nghiên cứu này tập trung vào bài toán đếm đối tượng trong điều kiện ít dữ liệu huấn luyện, trong đó đếm số lượng đối tượng mẫu (tức là được mô tả bởi một hoặc nhiều hình ảnh hỗ trợ) xuất hiện trong hình ảnh truy vấn. Thách thức lớn đặt ra là các đối tượng mục tiêu có thể được xếp chồng chất trong hình ảnh truy vấn, làm cho việc nhận diện từng đối tượng trở nên khó khăn. Để vượt qua khó khăn này, chúng tôi đề xuất một khối học mới, được trang bị một mô-đun so sánh độ tương đồng và một mô-đun tăng cường đặc trưng. Cụ thể, cho trước một hình ảnh hỗ trợ và một hình ảnh truy vấn, chúng tôi trước tiên tạo ra một bản đồ điểm số bằng cách so sánh các đặc trưng được chiếu của chúng ở mỗi vị trí không gian. Các bản đồ điểm số liên quan đến tất cả các hình ảnh hỗ trợ được thu thập lại và được chuẩn hóa trên cả chiều mẫu và các chiều không gian, tạo ra một bản đồ tương đồng đáng tin cậy. Sau đó, chúng tôi tăng cường đặc trưng của hình ảnh truy vấn bằng cách sử dụng các đặc trưng hỗ trợ dựa trên các hệ số trọng số tương tự đã phát triển. Thiết kế này khuyến khích mô hình xem xét hình ảnh truy vấn bằng cách tập trung nhiều hơn vào các vùng tương tự với hình ảnh hỗ trợ, dẫn đến đường biên rõ ràng hơn giữa các đối tượng khác nhau. Các thử nghiệm mở rộng trên các tập dữ liệu thử nghiệm và cấu hình huấn luyện khác nhau cho thấy chúng tôi vượt qua các phương pháp tiên tiến nhất một cách đáng kể. Ví dụ, trên tập dữ liệu FSC-147 quy mô lớn gần đây, chúng tôi cải thiện sai số tuyệt đối trung bình từ 22.08 xuống còn 14.32 (tăng 35%).

# Environments

```
timm
```


# Process

## 1. Dataset

- [dataset](https://github.com/pntrungbk15/TNVision/blob/main/task/objectcounting/fewshot/data/dataset.py)


## 2. Model Process 

- [model](https://github.com/pntrungbk15/TNVision/blob/main/task/objectcounting/fewshot/models/hlsafecount/model/hlsafecount.py)

<p align='center'>
    <img width='1500' src='assets/hlsafecount.png'>
</p>

# Run

```bash
python main.py --task_type objectcounting --model_type fewshot --model_name hlsafecount --yaml_config configs/objectcounting/fewshot/hlsafecount/custom.yaml
```

## Demo

### Kvasir
<p align="left">
  <img src=assets/kvasir.gif width="100%" />
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
|    | **Average**    |         98.37 |         98.31 |         95.53 |

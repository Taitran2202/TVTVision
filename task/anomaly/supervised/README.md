Để hiểu rõ hơn về precision và recall, hãy xem xét một ví dụ đơn giản về việc phân loại hình ảnh là "hình cọ" (positive) và "không phải hình cọ" (negative).

Giả sử chúng ta có một tập dữ liệu với 1000 hình ảnh:

- 900 hình ảnh là "không phải hình cọ" (true negative).
- 50 hình ảnh là "hình cọ" được dự đoán là "không phải hình cọ" (false negative).
- 30 hình ảnh là "không phải hình cọ" được dự đoán là "hình cọ" (false positive).
- 20 hình ảnh là "hình cọ" (true positive).

Precision (độ chính xác) là tỷ lệ giữa số lượng dự đoán đúng về "hình cọ" (true positive) trên tổng số hình ảnh được dự đoán là "hình cọ" (true positive + false positive):

Precision = True Positive / (True Positive + False Positive) = 20 / (20 + 30) ≈ 0.4

Recall (độ phục hồi, độ nhớ) là tỷ lệ giữa số lượng hình ảnh được dự đoán là "hình cọ" đúng (true positive) trên tổng số hình ảnh thật sự là "hình cọ" (true positive + false negative):

Recall = True Positive / (True Positive + False Negative) = 20 / (20 + 50) ≈ 0.2857

Precision đo lường khả năng của mô hình phân loại đúng các trường hợp dự đoán là "hình cọ". Càng cao, càng ít khả năng bị sai lệch. Trong trường hợp này, precision khá thấp (0.4) nghĩa là có một số trường hợp mô hình phân loại sai các hình ảnh là "không phải hình cọ" như "hình cọ".

Recall đo lường khả năng của mô hình phát hiện đúng các trường hợp "hình cọ". Càng cao, càng ít khả năng bỏ sót. Trong trường hợp này, recall khá thấp (0.2857) nghĩa là mô hình bỏ sót nhiều hình ảnh thật sự là "hình cọ" và phân loại sai chúng thành "không phải hình cọ".

Cần lưu ý rằng precision và recall thường có mối quan hệ đối nghịch. Khi bạn tăng precision, recall có thể giảm và ngược lại. Việc cân nhắc giữa precision và recall phụ thuộc vào yêu cầu và ứng dụng cụ thể của bài toán phân loại.
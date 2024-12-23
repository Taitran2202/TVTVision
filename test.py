import torch

# Giả sử bạn có một tensor chứa các hộp giới hạn [n, 4, 2]
boxes = torch.tensor([[[1, 2], [5, 2], [5, 6], [1, 6]],  # Box 1
                      [[3, 4], [7, 4], [7, 8], [3, 8]]]) # Box 2

# Tính toán xmin, ymin, xmax, ymax
xmin = boxes[:, :, 0].min(dim=1)[0]
ymin = boxes[:, :, 1].min(dim=1)[0]
xmax = boxes[:, :, 0].max(dim=1)[0]
ymax = boxes[:, :, 1].max(dim=1)[0]

# Tạo tensor mới chứa xmin, ymin, xmax, ymax
new_boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)

print(new_boxes)
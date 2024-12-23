import cv2
import numpy as np
import torchvision.transforms as transforms


def normalization(in_img):
    # convert ndarray into tensor
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    in_img = np.array(in_img, dtype=np.uint8)

    return image_transform(in_img)


def resize_aspect_ratio(image, resize):
    height, width = image.shape[:2]
    target_height, target_width = resize

    # Tính tỷ lệ giữa chiều cao và chiều rộng của ảnh gốc
    aspect_ratio = height / float(width)

    # Tính tỷ lệ giữa chiều cao và chiều rộng của ảnh mới
    target_aspect_ratio = target_height / float(target_width)

    # Nếu tỷ lệ giữa ảnh gốc và ảnh mới lớn hơn tỷ lệ giữa ảnh mới mong muốn
    if aspect_ratio > target_aspect_ratio:
        # Resize theo chiều cao
        new_height = target_height
        new_width = int(new_height / aspect_ratio)
    else:
        # Resize theo chiều rộng
        new_width = target_width
        new_height = int(new_width * aspect_ratio)

    # Resize ảnh về kích thước mới
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Tạo ảnh vuông kích thước (target_height, target_width) bằng cách đệm pixel bên phải và dưới
    top = 0
    bottom = target_height - new_height
    left = 0
    right = target_width - new_width
    resized_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=0)

    # Tính tỷ lệ scale giữa ảnh gốc và ảnh mới
    height_ratio = new_height / float(height)
    width_ratio = new_width / float(width)
    target_ratio = (height_ratio, width_ratio)

    valid_size_heatmap = (new_height // 2, new_width // 2)

    return resized_image, target_ratio, valid_size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

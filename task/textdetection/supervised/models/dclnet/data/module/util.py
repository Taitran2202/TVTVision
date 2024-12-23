import cv2
import numpy as np
import math


def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1, 2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x: (x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0, 8, 2):
        x, y = box[i], box[i+1]
        if [x, y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return new_box


def resize_img(img, vertices, min_side, max_side):
    resize_w = int(np.random.rand() * (max_side - min_side)) + min_side
    resize_h = int(np.random.rand() * (max_side - min_side)) + min_side
    h, w = img.shape[:2]

    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    for i, vertice in enumerate(vertices):
        for j in range(len(vertice)):
            if j % 2 != 0:
                vertice[j] *= resize_h / h
            else:
                vertice[j] *= resize_w / w

    return img, vertices


def crop_img(img, vertices, length, is_valid):
    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    start_w = int(np.random.rand() * remain_w)
    start_h = int(np.random.rand() * remain_h)
    box = (start_w, start_h, start_w + length, start_h + length)
    img_region = img.crop(box)

    for i, vertice in enumerate(vertices):
        for j in range(len(vertice)):
            if j % 2 != 0:
                vertice[j] -= start_h
            else:
                vertice[j] -= start_w
        if np.min(vertice) < 0 or np.max(vertice) > length:
            is_valid[i] = 0
    return img_region, vertices, is_valid


def crop_img(img, vertices, length, is_valid):
    # find random position
    remain_h = img.shape[0] - length
    remain_w = img.shape[1] - length
    start_w = int(np.random.rand() * remain_w)
    start_h = int(np.random.rand() * remain_h)
    box = (start_w, start_h, start_w + length, start_h + length)
    img_region = img[start_h:start_h+length, start_w:start_w+length]

    for i, vertice in enumerate(vertices):
        for j in range(len(vertice)):
            if j % 2 != 0:
                vertice[j] -= start_h
            else:
                vertice[j] -= start_w
        if np.min(vertice) < 0 or np.max(vertice) > length:
            is_valid[i] = 0
    return img_region, vertices, is_valid


def generate_label(img, vertices, is_valid, scale):
    label_height, label_width = int(
        np.around(img.shape[0] * scale)), int(np.around(img.shape[1] * scale))
    label_text, label_ignore, label_instance = [
        np.zeros((1, label_height, label_width), dtype=np.float32) for _ in range(3)]
    label_rho, label_theta = [
        np.zeros((4, label_height, label_width), dtype=np.float32) for _ in range(2)]
    for i, vertice in enumerate(vertices):
        temp = vertice*scale
        if is_valid[i] == 0:
            cv2.fillPoly(label_ignore[0, :, :], [
                         temp.reshape((-1, 2)).astype(np.int32)], 1)
            continue
        shrink_vertice = shrink_boundary(temp)
        cv2.fillPoly(label_text[0, :, :], [
                     shrink_vertice.reshape((-1, 2)).astype(np.int32)], 1)
        cv2.fillPoly(label_instance[0, :, :], [
                     shrink_vertice.reshape((-1, 2)).astype(np.int32)], i+1)
        r, c = np.where(label_instance[0, :, :] == i+1)
        for j in range(len(r)):
            label_rho[0, r[j], c[j]], label_theta[0, r[j], c[j]] = cal_rho_theta(
                temp[0]-c[j], r[j]-temp[1], temp[2]-c[j], r[j]-temp[3])
            label_rho[1, r[j], c[j]], label_theta[1, r[j], c[j]] = cal_rho_theta(
                temp[2]-c[j], r[j]-temp[3], temp[4]-c[j], r[j]-temp[5])
            label_rho[2, r[j], c[j]], label_theta[2, r[j], c[j]] = cal_rho_theta(
                temp[4]-c[j], r[j]-temp[5], temp[6]-c[j], r[j]-temp[7])
            label_rho[3, r[j], c[j]], label_theta[3, r[j], c[j]] = cal_rho_theta(
                temp[6]-c[j], r[j]-temp[7], temp[0]-c[j], r[j]-temp[1])
    return label_text, label_ignore, label_rho, label_theta


def cal_rho_theta(x1, y1, x2, y2):
    # AX+BY+C=0
    A = y2-y1
    B = x1-x2
    C = x2*y1 - x1*y2
    rho = abs(C) / np.sqrt(A**2 + B**2 + 1e-8)

    vector = (x1-x2, y1-y2) if y1 >= y2 else (x2-x1, y2-y1)
    cos_theta = vector[0] / np.sqrt(vector[0]**2 + vector[1]**2 + 1e-8)
    theta = np.arccos(cos_theta)  # math.pi
    theta = theta - math.pi/2 if theta > math.pi/2 else theta + math.pi/2
    if B*C > 0:
        theta += math.pi
    return rho, theta


def shrink_boundary(vertice, shrink_ratio=0.4):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertice
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4

    shrink_x1 = x1 + shrink_ratio*(center_x - x1)
    shrink_y1 = y1 + shrink_ratio*(center_y - y1)
    shrink_x2 = x2 + shrink_ratio*(center_x - x2)
    shrink_y2 = y2 + shrink_ratio*(center_y - y2)
    shrink_x3 = x3 + shrink_ratio*(center_x - x3)
    shrink_y3 = y3 + shrink_ratio*(center_y - y3)
    shrink_x4 = x4 + shrink_ratio*(center_x - x4)
    shrink_y4 = y4 + shrink_ratio*(center_y - y4)

    return np.array([shrink_x1, shrink_y1, shrink_x2, shrink_y2, shrink_x3, shrink_y3, shrink_x4, shrink_y4])

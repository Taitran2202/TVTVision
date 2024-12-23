import os
import cv2
import random
import numpy as np
import torch
from .watershed import segment_region_score
from utils.pre_processing import normalization


class PseudoCharBoxBuilder:
    def __init__(self, vis_test_dir: str, pseudo_vis_opt: bool, net, gaussian_builder, device: str = 'cpu'):
        self.vis_test_dir = vis_test_dir
        self.pseudo_vis_opt = pseudo_vis_opt
        self.gaussian_builder = gaussian_builder
        self.device = device
        if net is not None:
            self.net = net.eval()

    def crop_image_by_bbox(self, image, box, word):
        w = max(int(np.linalg.norm(box[0] - box[1])),
                int(np.linalg.norm(box[2] - box[3])))
        h = max(int(np.linalg.norm(box[0] - box[3])),
                int(np.linalg.norm(box[1] - box[2])))

        if w == 0:
            w = 1

        word_ratio = h / w
        # try:
        #     word_ratio = h / w
        # except:
        #     ipdb.set_trace()

        one_char_ratio = min(h, w) / (max(h, w) / len(word))
        if word_ratio > 1.7 or (word_ratio > 1.0 and one_char_ratio > 2.8):
            # if word_ratio > 2 or (word_ratio > 1.6 and one_char_ratio > 2.4):
            horizontal_text_bool = False
            long_side = h
            short_side = w
            M = cv2.getPerspectiveTransform(
                np.float32(box),
                np.float32(
                    np.array(
                        [
                            [long_side, 0],
                            [long_side, short_side],
                            [0, short_side],
                            [0, 0],
                        ]
                    )
                ),
            )
        else:
            horizontal_text_bool = True
            long_side = w
            short_side = h
            M = cv2.getPerspectiveTransform(
                np.float32(box),
                np.float32(
                    np.array(
                        [
                            [0, 0],
                            [long_side, 0],
                            [long_side, short_side],
                            [0, short_side],
                        ]
                    )
                ),
            )

        warped = cv2.warpPerspective(image, M, (long_side, short_side))

        return warped, M, horizontal_text_bool

    def inference_word_box(self, word_image):
        self.net.eval()

        with torch.no_grad():
            word_img_torch = normalization(word_image).unsqueeze(0)
            word_img_torch = word_img_torch.to(
                self.device, dtype=torch.float32)

            word_img_regions, _ = self.net(word_img_torch)

        return word_img_regions

    def clip_into_boundary(self, box, bound):
        if len(box) == 0:
            return box
        else:
            box[:, :, 0] = np.clip(box[:, :, 0], 0, bound[1])
            box[:, :, 1] = np.clip(box[:, :, 1], 0, bound[0])
            return box

    def get_confidence(self, real_len, pseudo_len):
        if pseudo_len == 0:
            return 0.0
        return (real_len - min(real_len, abs(real_len - pseudo_len))) / real_len

    def split_word_equal_gap(self, word_img_w, word_img_h, word):
        width = word_img_w
        height = word_img_h

        width_per_char = width / len(word)
        bboxes = []
        for j, char in enumerate(word):
            if char == " ":
                continue
            left = j * width_per_char
            right = (j + 1) * width_per_char
            bbox = np.array(
                [[left, 0], [right, 0], [right, height], [left, height]])
            bboxes.append(bbox)

        bboxes = np.array(bboxes, np.float32)
        return bboxes

    def get_angle(self, point, base_point):
        # Tính góc tạo bởi hai điểm
        delta_x = point[0] - base_point[0]
        delta_y = point[1] - base_point[1]
        return np.arctan2(delta_y, delta_x)

    def clockwise_sort(self, points):
        # Tìm điểm cơ sở (base_point) bằng cách tìm tọa độ nhỏ nhất theo trục y
        base_point = min(points, key=lambda p: p[1])

        # Sắp xếp các điểm dựa trên góc của chúng so với điểm cơ sở
        sorted_points = sorted(
            points, key=lambda p: self.get_angle(p, base_point))

        return np.array(sorted_points)

    def build_char_box(self, image, word_bbox, word, img_name=""):
        word_bbox = self.clockwise_sort(word_bbox)
        word_image, M, horizontal_text_bool = self.crop_image_by_bbox(
            image, word_bbox, word
        )
        real_word_without_space = word.replace("\s", "")
        real_char_len = len(real_word_without_space)

        scale = 128.0 / word_image.shape[0]

        word_image = cv2.resize(word_image, None, fx=scale, fy=scale)
        word_img_h, word_img_w, _ = word_image.shape

        scores = self.inference_word_box(word_image)
        region_score = scores[0, :, :].cpu().detach().numpy()
        region_score = np.uint8(np.clip(region_score, 0., 1.) * 255.)

        region_score_rgb = cv2.resize(region_score, (word_img_w, word_img_h))
        region_score_rgb = cv2.cvtColor(region_score_rgb, cv2.COLOR_GRAY2RGB)

        pseudo_char_bbox = segment_region_score(region_score)

        # Chỉ được sử dụng để trực quan hóa
        watershed_box = pseudo_char_bbox.copy()

        pseudo_char_bbox = self.clip_into_boundary(
            pseudo_char_bbox, region_score_rgb.shape
        )

        confidence = self.get_confidence(real_char_len, len(pseudo_char_bbox))

        if confidence <= 0.5:
            pseudo_char_bbox = self.split_word_equal_gap(
                word_img_w, word_img_h, word)
            confidence = 0.5

        if self.pseudo_vis_opt:
            self.visualize_pseudo_label(
                word_image, region_score, watershed_box, pseudo_char_bbox, img_name,
            )

        if len(pseudo_char_bbox) != 0:
            index = np.argsort(pseudo_char_bbox[:, 0, 0])
            pseudo_char_bbox = pseudo_char_bbox[index]

        pseudo_char_bbox /= scale

        M_inv = np.linalg.pinv(M)
        for i in range(len(pseudo_char_bbox)):
            pseudo_char_bbox[i] = cv2.perspectiveTransform(
                pseudo_char_bbox[i][None, :, :], M_inv
            )

        pseudo_char_bbox = self.clip_into_boundary(
            pseudo_char_bbox, image.shape)

        return pseudo_char_bbox, confidence, horizontal_text_bool

    def visualize_pseudo_label(
        self, word_image, region_score, watershed_box, pseudo_char_bbox, img_name,
    ):

        word_img_h, word_img_w, _ = word_image.shape
        word_img_cp1 = word_image.copy()
        word_img_cp2 = word_image.copy()
        _watershed_box = np.int32(watershed_box)
        _pseudo_char_bbox = np.int32(pseudo_char_bbox)

        region_score_color = cv2.applyColorMap(
            np.uint8(region_score), cv2.COLORMAP_JET)
        region_score_color = cv2.resize(
            region_score_color, (word_img_w, word_img_h))

        for box in _watershed_box:
            cv2.polylines(
                np.uint8(word_img_cp1),
                [np.reshape(box, (-1, 1, 2))],
                True,
                (255, 0, 0),
            )

        for box in _pseudo_char_bbox:
            cv2.polylines(
                np.uint8(word_img_cp2), [np.reshape(
                    box, (-1, 1, 2))], True, (255, 0, 0)
            )

        # LƯU Ý: Chỉ để trực quan hóa, hãy đặt bản đồ gaussian trên hộp char
        pseudo_gt_region_score = self.gaussian_builder.generate_region(
            word_img_h, word_img_w, [_pseudo_char_bbox], [True]
        )

        pseudo_gt_region_score = cv2.applyColorMap(
            (pseudo_gt_region_score * 255).astype("uint8"), cv2.COLORMAP_JET
        )

        overlay_img = cv2.addWeighted(
            word_image[:, :, ::-1], 0.7, pseudo_gt_region_score, 0.3, 5
        )

        vis_result = np.hstack(
            [
                word_image[:, :, ::-1],
                region_score_color,
                word_img_cp1[:, :, ::-1],
                word_img_cp2[:, :, ::-1],
                pseudo_gt_region_score,
                overlay_img,
            ]
        )

        # Lưu kết quả
        cv2.imwrite(
            os.path.join(self.vis_test_dir, "{}_{}".format(
                img_name, f"pseudo_char_bbox_{random.randint(0,100)}.jpg"
            ),
            ),
            vis_result
        )

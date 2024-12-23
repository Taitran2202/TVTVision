import cv2
import numpy as np
from .boxEnlarge import enlargebox


class GaussianBuilder(object):
    def __init__(self, init_size, sigma, enlarge_region, enlarge_affinity):
        self.init_size = init_size
        self.sigma = sigma
        self.enlarge_region = enlarge_region
        self.enlarge_affinity = enlarge_affinity
        self.gaussian_map = self.generate_gaussian_map()

    # def generate_gaussian_map(self):
    #     circle_mask = self.generate_circle_mask()

    #     x, y = np.meshgrid(np.arange(self.init_size),
    #                        np.arange(self.init_size))
    #     x = x - self.init_size / 2
    #     y = y - self.init_size / 2
    #     gaussian_map = np.exp(-(x**2 + y**2) / 2 /
    #                           (self.sigma ** 2)) / (2 * np.pi * (self.sigma ** 2))
    #     gaussian_map = gaussian_map * circle_mask
    #     gaussian_map = (gaussian_map / np.max(gaussian_map)).astype(np.float32)
    #     return gaussian_map

    # def generate_circle_mask(self):
    #     zero_arr = np.zeros((self.init_size, self.init_size), np.float32)
    #     circle_mask = cv2.circle(
    #         zero_arr,
    #         (self.init_size // 2, self.init_size // 2),
    #         self.init_size // 2,
    #         1,
    #         -1
    #     )
    #     return circle_mask

    # def four_point_transform(self, bbox):
    #     """
    #     Using the bbox, standard 2D gaussian map, returns Transformed 2d Gaussian map
    #     """
    #     width, height = (
    #         np.max(bbox[:, 0]).astype(np.int32),
    #         np.max(bbox[:, 1]).astype(np.int32),
    #     )

    #     init_points = np.array(
    #         [
    #             [0, 0],
    #             [self.init_size - 1, 0],
    #             [self.init_size - 1, self.init_size - 1],
    #             [0, self.init_size - 1],
    #         ],
    #         dtype="float32",
    #     )

    #     M = cv2.getPerspectiveTransform(init_points, bbox)
    #     warped_gaussian_map = cv2.warpPerspective(
    #         self.gaussian_map, M, (width, height))
    #     return warped_gaussian_map, width, height

    def generate_gaussian_map(self):
        circle_mask = self.generate_circle_mask()

        gaussian_map = np.zeros((self.init_size, self.init_size), np.float32)

        for i in range(self.init_size):
            for j in range(self.init_size):
                gaussian_map[i, j] = (
                    1
                    / 2
                    / np.pi
                    / (self.sigma ** 2)
                    * np.exp(
                        -1
                        / 2
                        * (
                            (i - self.init_size / 2) ** 2 / (self.sigma ** 2)
                            + (j - self.init_size / 2) ** 2 / (self.sigma ** 2)
                        )
                    )
                )

        gaussian_map = gaussian_map * circle_mask
        gaussian_map = (gaussian_map / np.max(gaussian_map)).astype(np.float32)

        return gaussian_map

    def generate_circle_mask(self):

        zero_arr = np.zeros((self.init_size, self.init_size), np.float32)
        circle_mask = cv2.circle(
            img=zero_arr,
            center=(self.init_size // 2, self.init_size // 2),
            radius=self.init_size // 2,
            color=1,
            thickness=-1,
        )

        return circle_mask

    def four_point_transform(self, bbox):
        """
        Using the bbox, standard 2D gaussian map, returns Transformed 2d Gaussian map
        """
        width, height = (
            np.max(bbox[:, 0]).astype(np.int32),
            np.max(bbox[:, 1]).astype(np.int32),
        )
        init_points = np.array(
            [
                [0, 0],
                [self.init_size, 0],
                [self.init_size, self.init_size],
                [0, self.init_size],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(init_points, bbox)
        warped_gaussian_map = cv2.warpPerspective(
            self.gaussian_map, M, (width, height))
        return warped_gaussian_map, width, height

    def add_gaussian_map_to_score_map(self, score_map, bbox, enlarge, is_horizontal, map_type=None):
        map_h, map_w = score_map.shape
        bbox = enlargebox(bbox, map_h, map_w, enlarge, is_horizontal)
        if np.any(bbox < 0) or np.any(bbox[:, 0] > map_w) or np.any(bbox[:, 1] > map_h):
            return score_map

        left, top = np.array(
            [np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        bbox -= (left, top)
        gaussian, w, h = self.four_point_transform(bbox.astype(np.float32))

        try:
            area = score_map[top: top + h, left: left + w]
            high_value = np.where(gaussian > area, gaussian, area)
            score_map[top: top + h, left: left + w] = high_value

        except Exception as e:
            print("Error : {}".format(e))

        return score_map

    def calculate_affinity_box_points(self, bbox_1, bbox_2, vertical=False):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        if vertical:
            tl = (bbox_1[0] + bbox_1[-1] + center_1) / 3
            tr = (bbox_1[1:3].sum(0) + center_1) / 3
            br = (bbox_2[1:3].sum(0) + center_2) / 3
            bl = (bbox_2[0] + bbox_2[-1] + center_2) / 3
        else:
            tl = (bbox_1[0:2].sum(0) + center_1) / 3
            tr = (bbox_2[0:2].sum(0) + center_2) / 3
            br = (bbox_2[2:4].sum(0) + center_2) / 3
            bl = (bbox_1[2:4].sum(0) + center_1) / 3
        affinity_box = np.array([tl, tr, br, bl]).astype(np.float32)
        return affinity_box

    def generate_region(
        self, img_h, img_w, word_level_char_bbox, horizontal_text_bools
    ):
        region_map = np.zeros([img_h, img_w], dtype=np.float32)
        for i in range(
            len(word_level_char_bbox)
        ):  # shape : [word_num, [char_num_in_one_word, 4, 2]]
            for j in range(len(word_level_char_bbox[i])):
                region_map = self.add_gaussian_map_to_score_map(
                    region_map,
                    word_level_char_bbox[i][j].copy(),
                    self.enlarge_region,
                    horizontal_text_bools[i],
                    map_type="region",
                )
        return region_map

    def generate_affinity(
        self, img_h, img_w, word_level_char_bbox, horizontal_text_bools
    ):

        affinity_map = np.zeros([img_h, img_w], dtype=np.float32)
        all_affinity_bbox = []
        for i in range(len(word_level_char_bbox)):
            for j in range(len(word_level_char_bbox[i]) - 1):
                affinity_bbox = self.calculate_affinity_box_points(
                    word_level_char_bbox[i][j], word_level_char_bbox[i][j + 1]
                )

                affinity_map = self.add_gaussian_map_to_score_map(
                    affinity_map,
                    affinity_bbox.copy(),
                    self.enlarge_affinity,
                    horizontal_text_bools[i],
                    map_type="affinity",
                )
                all_affinity_bbox.append(np.expand_dims(affinity_bbox, axis=0))

        if len(all_affinity_bbox) > 0:
            all_affinity_bbox = np.concatenate(all_affinity_bbox, axis=0)

        return affinity_map, all_affinity_bbox

import os
import cv2
import math
import random
import numpy as np
from glob import glob
from einops import rearrange
from typing import Tuple
import imgaug.augmenters as iaa
import torch
import torchvision.transforms as transforms


class AugTransforms(object):
    def __init__(self, datadir: str = None, target: str = None, resize: Tuple[int, int] = (256, 256),
                 texture_source_dir: str = None, trans_config=None, use_mask: bool = True,
                 bg_threshold: float = 100, bg_reverse: bool = False):
        # Mode
        self.datadir = datadir
        self.target = target
        self.trans_config = trans_config
        self.resize = resize

        # mask setting
        self.use_mask = use_mask
        self.bg_threshold = bg_threshold
        self.bg_reverse = bg_reverse

        self.file_list = None
        if self.datadir and self.target:
            image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
            self.file_list = [file for file in glob(os.path.join(
                self.datadir, self.target, 'images/*')) if os.path.splitext(file)[1].lower() in image_extensions]

        # load texture image file list
        if texture_source_dir:
            self.texture_source_file_list = glob(
                os.path.join(texture_source_dir, '*/*'))

        if trans_config:
            self.transparency_range = trans_config['transparency_range']
            self.perlin_scale = trans_config['perlin_scale']
            self.min_perlin_scale = trans_config['min_perlin_scale']
            self.perlin_noise_threshold = trans_config['perlin_noise_threshold']
            self.structure_grid_size = trans_config['structure_grid_size']

        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.anomaly_switch = False

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(
            np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    def __call__(self, image, target):
        # anomaly source
        if self.anomaly_switch and self.file_list and np.random.uniform() < 0.5:
            random_path = random.choice(self.file_list)
            image = cv2.imread(random_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(self.resize[1], self.resize[0]))

            file_extension = os.path.splitext(random_path)[1].lower()
            mask = cv2.imread(
                random_path.replace('images', 'masks').replace(
                    file_extension, f'_mask{file_extension}'),
                cv2.IMREAD_GRAYSCALE
            )
            mask = cv2.resize(mask, dsize=(
                self.resize[1], self.resize[0])).astype(bool).astype(int)

            target['mask'] = mask
            target['label'] = 1
            self.anomaly_switch = False

        elif self.anomaly_switch:
            image, mask = self.generate_anomaly(img=image)
            target['mask'] = mask
            target['label'] = 1
            self.anomaly_switch = False
        else:
            self.anomaly_switch = True

        image = self.img_transform(image)
        target['mask'] = torch.Tensor(target['mask']).to(torch.int64)

        return image, target

    def generate_anomaly(self, img):
        # step 1. generate mask
        # target foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(
                img=img)
        else:
            target_foreground_mask = np.ones(self.resize)

        # perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        # mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        # step 2. generate texture or structure anomaly

        # anomaly source
        anomaly_source_img = self.anomaly_source(img=img)

        # mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * \
            (mask_expanded * anomaly_source_img) + \
            (1 - factor) * (mask_expanded * img)

        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img

        return (anomaly_source_img.astype(np.uint8), mask)

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, target_background_mask = cv2.threshold(
            img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(
            bool).astype(int)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_background_mask = target_background_mask
        else:
            target_background_mask = 1 - target_background_mask

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale,
                              self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale,
                              self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = self._noise_rand_perlin_2d_np(
            (self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def _noise_rand_perlin_2d_np(self, shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]
            :delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

        def tile_grads(slice1, slice2): return np.repeat(np.repeat(
            gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)

        def dot(grad, shift): return (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])

        return math.sqrt(2) * self.lerp_np(self.lerp_np(n00, n10, t[..., 0]), self.lerp_np(n01, n11, t[..., 0]), t[..., 1])

    def lerp_np(self, x, y, w):
        fin_out = (y-x)*w + x
        return fin_out

    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform()
        if p < 0.5:
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.texture_source_file_list))
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(
            texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=(
            self.resize[1], self.resize[0])).astype(np.float32)

        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)

        assert self.resize[0] % self.structure_grid_size == 0
        grid_w = self.resize[1] // self.structure_grid_size
        grid_h = self.resize[0] // self.structure_grid_size

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=self.structure_grid_size,
            w=self.structure_grid_size
        ).astype(np.float32)

        return structure_source_img

import cv2
import numpy as np
import albumentations as A
from imgaug import augmenters as iaa
from numpy import ndarray, random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def strong_aug(p=0.5):
    return A.Compose([
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=p)


def remove_empty_boxes(boxes, labels):
    """Removes bounding boxes of W or H equal to 0 and its labels

    Args:
        boxes   (ndarray): NP Array with bounding boxes as lines
                           * BBOX[x1, y1, x2, y2]
        labels  (labels): Corresponding labels with boxes

    Returns:
        ndarray: Valid bounding boxes
        ndarray: Corresponding labels
    """
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)


class Transformer:
    """ image enhancement interface """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ Enhance the input image

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            Image, the image mode is RGB or HUV, the default RGB mode is no special instructions

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            bounding box

        label: `~np.ndarray` of shape `(n_objects, )`
            Category label

        Returns
        -------
        image, bbox, label:
            Enhanced data
        """
        raise NotImplementedError(
            "Image enhancement methods must be rewritten")


class Compose(Transformer):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.augment = strong_aug(p)

    def transform(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t.transform(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        augmented = self.augment(image=img.astype(np.uint8))
        return augmented["image"], boxes, labels


class ImageToFloat32(Transformer):
    def transform(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class BBoxToAbsoluteCoords(Transformer):
    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class BBoxToPercentCoords(Transformer):
    def transform(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ConvertColor(Transformer):
    def __init__(self, current: str, to: str):
        self.current = current
        self.to = to

    def transform(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.to == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.to == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class Resize(Transformer):
    def __init__(self, size=640):
        self.size = size

    def transform(self, image, boxes=None, labels=None):
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.size, self.size))
        # rescale bbox
        if boxes is not None:
            img_h, img_w = image.shape[:2]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] / orig_w * img_w
            boxes[..., [1, 3]] = boxes[..., [1, 3]] / orig_h * img_h

        return image, boxes, labels


class RandomSaturation(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(Transformer):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(Transformer):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle.transform(image)
        return image, boxes, labels


class RandomContrast(Transformer):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(Transformer):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def transform(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(Transformer):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def transform(self, image, boxes=None, labels=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = self.sample_options[random.randint(
                0, len(self.sample_options))]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(Transformer):
    def __init__(self, mean):
        self.mean = mean

    def transform(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(Transformer):
    def transform(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(Transformer):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def transform(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class ColorJitter(Transformer):
    """ color interference """

    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", to='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', to='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def transform(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness.transform(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort.transform(im, boxes, labels)
        return self.rand_light_noise.transform(im, boxes, labels)


class YoloAugmentation(Transformer):
    """ Data enhancer used in Yolo neural network training """

    def __init__(self, image_size=300, mean=(123, 117, 104), p=0.5):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            # ColorJitter(),
            # Expand(mean),
            RandomSampleCrop(),
            RandomMirror(),
            Resize(image_size)
        ], p=p)

    def transform(self, image, bbox, label):
        """ image enhancement


        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB image

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            bounding box

        label: `~np.ndarray` of shape `(n_objects, )`
            Category label

        Returns
        -------
        image, bbox, label:
            Enhanced data
        """
        return self.transformers.transform(image, bbox, label)


class YoloColorAugmentation(Transformer):
    """ Data enhancer used when Yolo neural network training uses mosaic augmentation """

    def __init__(self, image_size=416, mean=(123, 117, 104), p=0.5):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.p = p
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            # ColorJitter(),
            RandomMirror(),
            Resize(image_size),
        ], p=p)

    def transform(self, image, bbox, label):
        return self.transformers.transform(image, bbox, label)


class DETRAugmentation(Transformer):
    """ Data enhancer used in DETR neural network training """

    def __init__(self, image_size=300, mean=(123, 117, 104), p=0.5):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            # ColorJitter(),
            # Expand(mean),
            RandomSampleCrop(),
            RandomMirror(),
            Resize(image_size),
            BBoxToPercentCoords()
        ], p=p)

    def transform(self, image, bbox, label):
        """ image enhancement


        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB image

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            bounding box

        label: `~np.ndarray` of shape `(n_objects, )`
            Category label

        Returns
        -------
        image, bbox, label:
            Enhanced data
        """
        return self.transformers.transform(image, bbox, label)


class DETRColorAugmentation(Transformer):
    """ Data enhancer used when DETR neural network training uses mosaic augmentation """

    def __init__(self, image_size=416, mean=(123, 117, 104), p=0.5):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.p = p
        self.transformers = Compose([
            ImageToFloat32(),
            BBoxToAbsoluteCoords(),
            # ColorJitter(),
            RandomSampleCrop(),
            RandomMirror(),
            Resize(image_size),
            BBoxToPercentCoords(),
        ], p=p)

    def transform(self, image, bbox, label):
        return self.transformers.transform(image, bbox, label)

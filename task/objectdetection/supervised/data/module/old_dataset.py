# import os
# import cv2
# from glob import glob
# import numpy as np
# import albumentations as A
# import xml.etree.ElementTree as ET
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms


# class VOCAnnotationTransform(object):
#     """Transforms a VOC annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes
#     Arguments:
#         class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#             (default: alphabetic indexing of VOC's 20 classes)
#         keep_difficult (bool, optional): keep difficult instances or not
#             (default: False)
#         height (int): height
#         width (int): width
#     """

#     def __init__(self, class_names=None, keep_difficult=False):
#         self.class_to_ind = {k: v for v, k in enumerate(class_names)}
#         self.keep_difficult = keep_difficult

#     def __call__(self, target):
#         """
#         Arguments:
#             target (annotation) : the target annotation to be made usable
#                 will be an ET.Element
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class name]
#         """
#         res = []
#         for obj in target.iter('object'):
#             difficult = int(obj.find('difficult').text) == 1
#             if not self.keep_difficult and difficult:
#                 continue
#             name = obj.find('name').text.lower().strip()
#             bbox = obj.find('bndbox')

#             pts = ['xmin', 'ymin', 'xmax', 'ymax']
#             bndbox = []
#             for i, pt in enumerate(pts):
#                 cur_pt = int(float(bbox.find(pt).text)) - 1
#                 # scale height or width
#                 cur_pt = cur_pt if i % 2 == 0 else cur_pt
#                 bndbox.append(cur_pt)
#             label_idx = self.class_to_ind[name]
#             bndbox.append(label_idx)
#             res += [bndbox]  # [x1, y1, x2, y2, label_ind]

#         return res  # [[x1, y1, x2, y2, label_ind], ... ]


# class YOLODataset(Dataset):
#     """VOC Detection Dataset Object
#     input is image, target is annotation
#     Arguments:
#         root (string): filepath to VOCdevkit folder.
#         image_set (string): imageset to use (eg. 'train', 'val', 'test')
#         transform (callable, optional): transformation to perform on the
#             input image
#         target_transform (callable, optional): transformation to perform on the
#             target `annotation`
#             (eg: take in caption string, return tensor of word indices)
#         dataset_name (string, optional): which dataset to load
#             (default: 'VOC2007')
#     """

#     def __init__(self, datadir: str, target: str, is_train: bool, class_names,
#                  resize: int = 640, keep_difficult: bool = False, trans_config=None):
#         super(YOLODataset, self).__init__()

#         # data
#         self.datadir = datadir
#         self.target = target
#         self.class_names = class_names
#         self.resize = resize
#         self.keep_difficult = keep_difficult
#         self.trans_config = trans_config

#         # mode
#         self.is_train = is_train
#         self.target_transform = VOCAnnotationTransform(
#             class_names, keep_difficult)

#         image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
#         self.file_list = [file for file in glob(os.path.join(
#             self.datadir, self.target, 'train/*' if is_train else 'test/*')) if os.path.splitext(file)[1].lower() in image_extensions]

#         # Define augmentation pipeline
#         train_transforms = [
#             A.LongestMaxSize(max_size=int(self.resize * 1.1)),
#             A.PadIfNeeded(
#                 min_height=int(self.resize * 1.1),
#                 min_width=int(self.resize * 1.1),
#                 border_mode=cv2.BORDER_CONSTANT,
#             ),
#             A.RandomCrop(width=self.resize, height=self.resize),
#         ]

#         if self.trans_config['color_distortion']['option']:
#             train_transforms.append(
#                 A.ColorJitter(
#                     brightness=self.trans_config['color_distortion']['brightness'],
#                     contrast=self.trans_config['color_distortion']['contrast'],
#                     saturation=self.trans_config['color_distortion']['saturation'],
#                     hue=self.trans_config['color_distortion']['hue'],
#                     p=self.trans_config['color_distortion']['prob']
#                 )
#             )

#         if self.trans_config['translation']['option']:
#             train_transforms.append(
#                 A.OneOf(
#                     [
#                         A.ShiftScaleRotate(
#                             rotate_limit=self.trans_config['translation']['rotate_limit'], p=self.trans_config[
#                                 'translation']['p'], border_mode=cv2.BORDER_CONSTANT
#                         ),
#                         A.IAAAffine(
#                             shear=self.trans_config['translation']['shear'], p=self.trans_config['translation']['p'], mode="constant"),
#                     ],
#                     p=self.trans_config['translation']['prob'],
#                 ),
#             )

#         if self.trans_config['horizontal_flip']['option']:
#             train_transforms.append(
#                 A.HorizontalFlip(
#                     p=self.trans_config['horizontal_flip']['prob'])
#             )

#         if self.trans_config['blur']['option']:
#             train_transforms.append(
#                 A.Blur(p=self.trans_config['blur']['prob'])
#             )
#         if self.trans_config['clahe']['option']:
#             train_transforms.append(
#                 A.CLAHE(p=self.trans_config['clahe']['prob'])
#             )

#         if self.trans_config['posterize']['option']:
#             train_transforms.append(
#                 A.Posterize(p=self.trans_config['posterize']['prob'])
#             )

#         if self.trans_config['togray']['option']:
#             train_transforms.append(
#                 A.ToGray(p=self.trans_config['togray']['prob'])
#             )

#         if self.trans_config['channelshuffle']['option']:
#             train_transforms.append(
#                 A.ChannelShuffle(p=self.trans_config['channelshuffle']['prob'])
#             )

#         # Define image transformation pipeline
#         self.train_transform = A.Compose(
#             train_transforms,
#             bbox_params=A.BboxParams(
#                 format="pascal_voc", min_area=1, min_visibility=self.trans_config['min_visibility'], label_fields=['class_labels'])
#         )

#         # convert to tensor
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         image, target = self.pull_item(index)
#         return image, target

#     def pull_item(self, index):
#         if self.is_train:
#             image, target = self.load_image_target(index)
#             augmentations = self.train_transform(
#                 image=image, bboxes=target['boxes'], class_labels=target['labels'])
#             image = augmentations["image"]
#             target['boxes'] = augmentations["bboxes"]
#             target['labels'] = augmentations["class_labels"]

#             return self.transform(image), target
#         else:
#             img_path, target = self.load_test_dataset(index)

#             return img_path, target

#     def load_image_target(self, index):
#         file_path = self.file_list[index]

#         # load an image
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         height, width, _ = image.shape

#         # load an annotation
#         anno = ET.parse(file_path.replace('JPEGImages', 'Annotations').replace(
#             '.jpg', '.xml')).getroot()

#         if self.target_transform is not None:
#             anno = self.target_transform(anno)

#         # guard against no boxes via resizing
#         anno = np.array(anno).reshape(-1, 5)
#         target = {
#             "boxes": anno[:, :4],
#             "labels": anno[:, 4],
#             "orig_size": [height, width]
#         }

#         return image, target

#     def load_test_dataset(self, index):
#         img_path = self.file_list[index]

#         # laod an annotation
#         anno = ET.parse(img_path.replace('JPEGImages', 'Annotations').replace(
#             '.jpg', '.xml')).getroot()

#         if self.target_transform is not None:
#             anno = self.target_transform(anno)

#         # guard against no boxes via resizing
#         anno = np.array(anno).reshape(-1, 5)
#         target = {
#             "boxes": anno[:, :4],
#             "labels": anno[:, 4]
#         }

#         return img_path, target
    
    
    
    
    
    
# import os
# import cv2
# import random
# from glob import glob
# import numpy as np
# import xml.etree.ElementTree as ET
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# from .module import ColorAugmentation, YoloAugmentation


# class VOCAnnotationTransform(object):
#     """Transforms a VOC annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes
#     Arguments:
#         class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#             (default: alphabetic indexing of VOC's 20 classes)
#         keep_difficult (bool, optional): keep difficult instances or not
#             (default: False)
#         height (int): height
#         width (int): width
#     """

#     def __init__(self, class_names=None, keep_difficult=False):
#         self.class_to_ind = {k: v for v, k in enumerate(class_names)}
#         self.keep_difficult = keep_difficult

#     def __call__(self, target):
#         """
#         Arguments:
#             target (annotation) : the target annotation to be made usable
#                 will be an ET.Element
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class name]
#         """
#         res = []
#         for obj in target.iter('object'):
#             difficult = int(obj.find('difficult').text) == 1
#             if not self.keep_difficult and difficult:
#                 continue
#             name = obj.find('name').text.lower().strip()
#             bbox = obj.find('bndbox')

#             pts = ['xmin', 'ymin', 'xmax', 'ymax']
#             bndbox = []
#             for i, pt in enumerate(pts):
#                 cur_pt = int(float(bbox.find(pt).text)) - 1
#                 # scale height or width
#                 cur_pt = cur_pt if i % 2 == 0 else cur_pt
#                 bndbox.append(cur_pt)
#             label_idx = self.class_to_ind[name]
#             bndbox.append(label_idx)
#             res += [bndbox]  # [x1, y1, x2, y2, label_ind]

#         return res  # [[x1, y1, x2, y2, label_ind], ... ]


# class YOLODataset(Dataset):
#     """VOC Detection Dataset Object
#     input is image, target is annotation
#     Arguments:
#         root (string): filepath to VOCdevkit folder.
#         image_set (string): imageset to use (eg. 'train', 'val', 'test')
#         transform (callable, optional): transformation to perform on the
#             input image
#         target_transform (callable, optional): transformation to perform on the
#             target `annotation`
#             (eg: take in caption string, return tensor of word indices)
#         dataset_name (string, optional): which dataset to load
#             (default: 'VOC2007')
#     """

#     def __init__(self, datadir: str, target: str, is_train: bool, class_names, resize: int = 640, keep_difficult: bool = False,  use_mosaic=False, use_mixup=False):
#         super(YOLODataset, self).__init__()

#         # data
#         self.datadir = datadir
#         self.target = target
#         self.class_names = class_names
#         self.resize = resize
#         self.keep_difficult = keep_difficult

#         # mode
#         self.is_train = is_train
#         self.target_transform = VOCAnnotationTransform(
#             class_names, keep_difficult)

#         image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
#         self.file_list = [file for file in glob(os.path.join(
#             self.datadir, self.target, 'train/*' if is_train else 'test/*')) if os.path.splitext(file)[1].lower() in image_extensions]

#         # augmentation
#         self.use_mosaic = use_mosaic
#         self.use_mixup = use_mixup

#         # convert to tensor
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         image, target = self.pull_item(index)

#         return image, target

#     def make_mosaic(self, index: int):
#         """ Create mosaic enhanced images """
#         # Randomly select three images
#         indexes = list(range(len(self.file_list)))
#         choices = random.sample(indexes[:index]+indexes[index+1:], 3)
#         choices.append(index)

#         # Read in the four pictures and their labels used to make the mosaic
#         images, bboxes, labels = [], [], []
#         for i in choices:
#             image, bbox, label = self.load_image_target(i)
#             images.append(image)
#             bboxes.append(bbox)
#             labels.append(label)

#         # Create a mosaic image and select splicing points
#         img_size = self.resize
#         mean = np.array([123, 117, 104])
#         mosaic_img = np.ones((img_size*2, img_size*2, 3))*mean
#         xc = int(random.uniform(img_size//2, 3*img_size//2))
#         yc = int(random.uniform(img_size//2, 3*img_size//2))

#         # Stitch images
#         for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):
#             # Preserve/not preserve proportion scaling image
#             ih, iw, _ = image.shape

#             s = np.random.choice(np.arange(50, 210, 10))/100
#             if np.random.randint(2):
#                 r = img_size / max(ih, iw)
#                 if r != 1:
#                     interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
#                     image = cv2.resize(
#                         image, (int(iw*r*s), int(ih*r*s)), interpolation=interp)
#             else:
#                 image = cv2.resize(image, (int(img_size*s), int(img_size*s)))

#             # Paste the image to the upper left corner, upper right corner, lower left corner and lower right corner of the splicing point
#             h, w, _ = image.shape
#             if i == 0:
#                 # The coordinates of the upper left corner and lower right corner of the pasted part in the mosaic image
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
#                 # The coordinates of the upper left corner and lower right corner of the pasted part in the original image
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
#             elif i == 1:
#                 x1a, y1a, x2a, y2a = xc, max(
#                     yc - h, 0), min(xc + w, img_size * 2), yc
#                 x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#             elif i == 2:
#                 x1a, y1a, x2a, y2a = max(
#                     xc - w, 0), yc, xc, min(img_size * 2, yc + h)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#             elif i == 3:
#                 x1a, y1a, x2a, y2a = xc, yc, min(
#                     xc + w, img_size * 2), min(img_size * 2, yc + h)
#                 x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#             mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

#             # translate the coordinates
#             dx = x1a - x1b
#             dy = y1a - y1b
#             bbox[:, [0, 2]] = (w * bbox[:, [0, 2]] / iw) + dx
#             bbox[:, [1, 3]] = (h * bbox[:, [1, 3]] / ih) + dy

#         # Handle bounding boxes beyond the coordinate system of the mosaic image
#         bbox = np.clip(np.vstack(bboxes), 0, 2*img_size)
#         label = np.hstack(labels)

#         # Remove bounding boxes that are too small
#         bbox_w = bbox[:, 2] - bbox[:, 0]
#         bbox_h = bbox[:, 3] - bbox[:, 1]
#         # np.logical_and(bbox_w > 3, bbox_h > 3)
#         mask = np.logical_and(bbox_w > 5, bbox_h > 5)
#         bbox, label = bbox[mask], label[mask]
#         if len(bbox) == 0:
#             bbox = np.zeros((1, 4))
#             label = np.array([0])


#         return mosaic_img, bbox, label

#     def pull_item(self, index):
#         if self.is_train:
#             if self.use_mosaic and np.random.randint(2):
#                 image, bbox, label = self.make_mosaic(index)

#                 # mixup
#                 if self.use_mixup and np.random.randint(2):
#                     index_ = np.random.randint(0, len(self))
#                     image_, bbox_, label_ = self.make_mosaic(index_)
#                     r = np.random.beta(8, 8)
#                     image = (image*r+image_*(1-r)).astype(np.uint8)
#                     bbox = np.vstack((bbox, bbox_))
#                     label = np.hstack((label, label_))

#                 # Image enhancement
#                 image, bbox, label = ColorAugmentation(
#                     self.resize, p=0.5).transform(image, bbox, label)
#             else:
#                 image, bbox, label = self.load_image_target(index)
#                 image, bbox, label = YoloAugmentation(
#                     self.resize, p=0.5).transform(image, bbox, label)

#             height, width, _ = image.shape
#             target = {
#                 "boxes": bbox,
#                 "labels": label,
#                 "orig_size": [height, width]
#             }
#             return self.transform(image), target
#         else:
#             img_path, target = self.load_test_dataset(index)

#             return img_path, target

#     def load_image_target(self, index):
#         file_path = self.file_list[index]

#         # load an image
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # load an annotation
#         anno = ET.parse(file_path.replace('JPEGImages', 'Annotations').replace(
#             '.jpg', '.xml')).getroot()

#         if self.target_transform is not None:
#             anno = self.target_transform(anno)

#         # guard against no boxes via resizing
#         anno = np.array(anno).reshape(-1, 5)
#         boxes = anno[:, :4]
#         labels = anno[:, 4]

#         return image, boxes, labels

#     def load_test_dataset(self, index):
#         img_path = self.file_list[index]

#         # laod an annotation
#         anno = ET.parse(img_path.replace('JPEGImages', 'Annotations').replace(
#             '.jpg', '.xml')).getroot()

#         if self.target_transform is not None:
#             anno = self.target_transform(anno)

#         # guard against no boxes via resizing
#         anno = np.array(anno).reshape(-1, 5)
#         target = {
#             "boxes": anno[:, :4],
#             "labels": anno[:, 4]
#         }

#         return img_path, target

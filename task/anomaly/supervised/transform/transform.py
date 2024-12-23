import torch
import torchvision.transforms as transforms


class Transforms(object):
    def __init__(self):
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, image, target):
        image = self.img_transform(image)
        target['mask'] = torch.Tensor(target['mask']).to(torch.int64)

        return image, target

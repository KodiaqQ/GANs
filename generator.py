import os

import cv2
from albumentations import *
from albumentations import pytorch as AT
from torch.utils.data import Dataset


class Pipes(Dataset):
    def __init__(self,
                 options):
        super().__init__()
        self.path = options['path']
        self.size = options['size']
        self.images = os.listdir(self.path)
        print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        interpolation = cv2.INTER_AREA

        image_path = os.path.join(self.path, self.images[index])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (image.shape[0] < self.size[0]) or (image.shape[1] < self.size[1]):
            interpolation = cv2.INTER_CUBIC

        return self.transform(image, interpolation)

    def transform(self, image, interpolation):
        augs = Compose([
            OneOf([
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90()
            ]),
            RandomBrightnessContrast(p=0.25),
            Resize(self.size[0], self.size[1], interpolation=interpolation),
            Normalize(),
            AT.ToTensor()
        ], p=1)

        input = {'image': image}
        output = augs(**input)
        return output['image']

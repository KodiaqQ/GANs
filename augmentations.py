import os
from albumentations import *
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder with sample data')
parser.add_argument('--iter', help='Number of albumentations iterations')
parser.add_argument('--target', help='Target folder for new data')
args = parser.parse_args()


if __name__ == '__main__':
    images = os.listdir(args.folder)[:10]

    augs = Compose([
        CenterCrop(height=560, width=560),
        OneOf([
            HorizontalFlip(),
            Rotate(limit=45),
            VerticalFlip(),
            RandomRotate90()
        ]),
        OneOf([
            RandomBrightnessContrast(),
            RandomGamma()
        ])
    ])
    k = 0

    for i, path in enumerate(images):
        image = cv2.imread(os.path.join(args.folder, path), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('running %s image', i)

        for j in range(int(args.iter)):
            sample = {'image': image}
            output = augs(**sample)

            cv2.imwrite(os.path.join(args.target, str(k) + '.jpg'), output['image'])
            k += 1
    print('done')

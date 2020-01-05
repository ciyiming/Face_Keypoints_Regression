import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import time


kps = ['kp%s%s' % (e, i) for i in range(1, 22) for e in ['x', 'y']]


class Resize:
    def __init__(self,  height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, landmarks, cls = sample['image'], sample['landmarks'], sample['cls']
        if type(image) is np.ndarray:
            img_w, img_h = image.shape
        else:
            img_w, img_h = image.size
        h_ratio = self.height / img_h
        w_ratio = self.width / img_w
        image_resize = np.asarray(image.resize((self.width, self.height), Image.BILINEAR), dtype=np.float32)
        if cls:
            landmarks = np.array([landmarks[i] * w_ratio if i % 2 == 0
                                  else landmarks[i] * h_ratio for i in range(len(landmarks))], dtype=np.float32)
        return {'image': image_resize, 'landmarks': landmarks, 'cls': cls}


class Normalize:
    def __call__(self, sample):
        image, landmarks, cls = sample['image'], sample['landmarks'], sample['cls']
        image = self.channel_norm(np.asarray(image, dtype=np.float32))
        landmarks = np.array(landmarks, dtype=np.float32)
        return {'image': image, 'landmarks': landmarks, 'cls': cls}

    def channel_norm(self, img):
        mean = np.mean(img)
        std = np.std(img)
        pixels = (img - mean) / (std + 0.0000001)
        return pixels


class ToTensor:
    def __call__(self, sample):
        image, landmarks, cls = sample['image'], sample['landmarks'], sample['cls']
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.float32))),
                'landmarks': torch.from_numpy(np.ascontiguousarray(np.array(landmarks, dtype=np.float32))), 'cls': cls}


class RandomHorizontalFlip:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        image, landmarks, cls = sample['image'], sample['landmarks'], sample['cls']
        if type(image) is np.ndarray:
            img_w, img_h = image.shape
        else:
            img_w, img_h = image.size
        flip = np.random.choice([0, 1], None, p=[1 - self.ratio, self.ratio], replace=True)
        if flip:
            image = np.flip(np.asarray(image, dtype=np.float32), 1)
            if cls:
                landmarks = np.array([img_w + 1 - landmarks[i] if i % 2 == 0
                                       else landmarks[i] for i in range(len(landmarks))], dtype=np.float32)
        return {'image': image, 'landmarks': landmarks, 'cls': cls}


class FaceLandmarksDataset(Dataset):
    def __init__(self, file_dir, label_file, transform=None):
        self.file_dir = file_dir
        self.label_file = label_file
        self.transform = transform
        if not os.path.isfile(self.label_file):
            print(self.label_file + 'does not exist!')
        self.file_info = pd.read_csv(label_file)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # get image
        image_path = self.file_info['file'][idx]
        rect = self.file_info.loc[idx, ['x1', 'y1', 'x2', 'y2']]
        cls = int(self.file_info['class'][idx])
        if cls:
            landmarks = self.file_info.loc[idx, kps]
        else:
            landmarks = np.zeros(42)
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        image = Image.open(image_path).convert('L')
        img_crop = image.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)
        sample = {'image': img_crop, 'landmarks': landmarks, 'cls': cls}
        if self.transform:
            sample = self.transform(sample)
        return sample


data_transforms = {
    'train': transforms.Compose([
        Resize(112, 112),
        Normalize(),
        RandomHorizontalFlip(),
        ToTensor()
    ]),
    'test': transforms.Compose([
        Resize(112, 112),
        Normalize(),
        ToTensor()
    ])
}

if __name__ == '__main__':
    train_set = FaceLandmarksDataset('./', 'train.csv', transform=data_transforms['train'])
    for i in range(0, len(train_set)):
        sample = train_set[i]
        img = sample['image']
        landmarks = sample['landmarks']
        cls = sample['cls']
        if cls:
            landmarks = landmarks.reshape(-1, 2).tolist()
            x, y = [l[0] for l in landmarks], [l[1] for l in landmarks]
            plt.scatter(x, y, c='r')
        plt.imshow(img.squeeze(0))
        plt.show()
        time.sleep(0.1)
        plt.close('all')

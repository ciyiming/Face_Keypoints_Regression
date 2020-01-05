import os
import torch
import linecache
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ResizeNormalize:
    def __init__(self,  height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img_w, img_h = image.size
        h_ratio = self.height / img_h
        w_ratio = self.width / img_w
        image_resize = np.asarray(image.resize((self.width, self.height), Image.BILINEAR), dtype=np.float32)
        image = self.channel_norm(image_resize)
        landmarks = np.array([landmarks[i] * w_ratio if i % 1
                              else landmarks[i] * h_ratio for i in range(len(landmarks))])
        return {'image': image, 'landmarks': landmarks}

    def channel_norm(self, img):
        mean = np.mean(img)
        std = np.std(img)
        pixels = (img - mean) / (std + 0.0000001)
        return pixels


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    def __init__(self, file_dir, label_file, transform=None):
        self.file_dir = file_dir
        self.label_file = label_file
        self.transform = transform
        if not os.path.isfile(self.label_file):
            print(self.label_file + 'does not exist!')
        self.file_info = open(os.path.join(self.file_dir, self.label_file))
        self.size = len(self.file_info.readlines())

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # get image
        line = linecache.getline(os.path.join(self.file_dir, self.label_file), idx+1)
        img_name, rect, landmarks = self.parse_line(line)
        if not os.path.isfile(img_name):
            print(img_name + ' does not exist!')
            return None
        image = Image.open(img_name).convert('L')
        img_crop = image.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)
        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample

    def parse_line(self, line):
        line_parts = line.strip().split()
        img_name = line_parts[0]
        rect = list(map(int, list(map(float, line_parts[1:5]))))
        landmarks = list(map(float, line_parts[5: len(line_parts)]))
        return img_name, rect, landmarks


data_transforms = {
    'train': transforms.Compose([
        ResizeNormalize(112, 112),
        ToTensor()
    ]),
    'test': transforms.Compose([
        ResizeNormalize(112, 112),
        ToTensor()
    ])
}

if __name__ == '__main__':
    train_set = FaceLandmarksDataset('./', 'train.txt', transform=data_transforms['train'])
    for i in range(1, len(train_set)):
        sample = train_set[i]
        img = sample['image']
        landmarks = sample['landmarks']
        landmarks = landmarks.reshape(-1, 2).tolist()
        x, y = [l[0] for l in landmarks], [l[1] for l in landmarks]
        plt.imshow(transforms.ToPILImage()(img.squeeze(0)))
        plt.scatter(x, y, c='r')
        plt.show()

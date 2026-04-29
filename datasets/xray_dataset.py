import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class XRayDataset(Dataset):
    """X 光图像分割数据集

    Args:
        images_path_list: 图像文件路径列表
        labels_path_list: 标签文件路径列表
        split: 'Train' 或 'Test'，训练模式下 epoch 长度固定为 1600
        augmentation: 是否启用随机翻转增强
        device: 张量存放设备
    """

    def __init__(self, images_path_list, labels_path_list, split='Train',
                 augmentation=False, device='cpu'):
        self.images = images_path_list
        self.labels = labels_path_list
        self.augmentation = augmentation
        self.device = device
        self.split = split

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        if self.augmentation:
            self.same_augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        if self.split == 'Train':
            self._getitem = self._getitem_train
            self.len_data = 100 * 16
        else:
            self._getitem = self._getitem_test
            self.len_data = len(self.images)

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        return self._getitem(idx)

    def _getitem_test(self, idx):
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)
        label = 1.0 * (label != 0)

        return {'rgb': image, 'label': label, 'fname': name}

    def _getitem_train(self, idx):
        idx = idx % len(self.images)
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        if self.augmentation:
            seed = np.random.randint(0, 10000)
            torch.random.manual_seed(seed)
            image = self.same_augmentation(image)
            label = self.same_augmentation(label)
            torch.random.manual_seed(seed)

        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)
        label = 1.0 * (label != 0)

        return {'rgb': image, 'label': label, 'fname': name}

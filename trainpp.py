import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from network import U_Net_PP  # 确保引入新的U-Net++模型
import os
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


class XRayDataset(Dataset):

    def __init__(self, images_path_list, labels_path_list, split='Train', augmentation=True, device='cuda:1',
                 image_size=(512, 512)):
        self.images = images_path_list
        self.labels = labels_path_list
        self.augmentation = augmentation
        self.device = device
        self.split = split

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        if self.augmentation:
            self.same_augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ])

        if self.split == 'Train':
            self._getitem = self._getitem_train
            self.len_data = 100 * 16
        else:
            self._getitem = self._getitem_test
            self.len_data = len(self.images)

    def __getitem__(self, idx):
        return self._getitem(idx)

    def _getitem_test(self, idx):
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)
        label = 1. * (label != 0)

        return {'rgb': image,
                'label': label,
                'fname': name}

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
        label = 1. * (label != 0)

        return {'rgb': image,
                'label': label,
                'fname': name}

    def __len__(self):
        return self.len_data


image_path_train = './dataset/train/xray/'
labels_path_train = './dataset/train/mask/'

# 加载图像和标签文件名
image_names_train = [filename for filename in os.listdir(image_path_train)]
label_names_train = [filename for filename in os.listdir(labels_path_train)]

# 创建完整的文件路径
train_image_path = [os.path.join(image_path_train, file_name) for file_name in image_names_train]
train_mask_path = [os.path.join(labels_path_train, file_name) for file_name in label_names_train]

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据集和数据加载器
train_dataset = XRayDataset(
    images_path_list=train_image_path,
    labels_path_list=train_mask_path,
    augmentation=False,
    split='Train',
    device=device
)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义U-Net++模型
unetpp = U_Net_PP().to(device)

unetpp.train()

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(unetpp.parameters(), lr=0.001)

# 训练循环
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader), 0):
        inputs, labels = data['rgb'].to(device), data['label'].to(device)

        optimizer.zero_grad()

        outputs = unetpp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

print('训练完成')
torch.save({'state_dict': unetpp.state_dict()}, './checkpoint/UNET_model_MY.pth')

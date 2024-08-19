import torch
from matplotlib import pyplot as plt
import cv2  # 导入OpenCV库，用于图像处理
from network import U_Net_PP  # 导入自定义的U_Net网络模型
import numpy as np
from PIL import Image  # 导入PIL库中的Image模块
import torchvision.transforms as transforms  # 导入torchvision.transforms模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch中的Dataset和DataLoader模块
import os  # 导入os库
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

# 创建一个U_Net的实例
unet = U_Net_PP()


# 加载分类网络预训练参数
def load_networks(model, path):
    net = model
    state_dict = torch.load(path)  # 加载预训练参数
    print('loading the model from %s' % (path))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net_state = net.state_dict()  # 获取模型的状态字典（包含模型的权重和偏置）
    is_loaded = {n: False for n in net_state.keys()}  # 创建一个字典，用于跟踪加载的参数
    for name, param in state_dict['state_dict'].items():  # 遍历预训练参数字典
        if name in net_state:  # 如果模型中存在对应的参数
            try:
                net_state[name].copy_(param)  # 将预训练参数复制到模型中
                is_loaded[name] = True  # 标记该参数已加载
            except Exception:
                print('While copying the parameter named [%s], '
                      'whose dimensions in the model are %s and '
                      'whose dimensions in the checkpoint are %s.'
                      % (name, list(net_state[name].shape),
                         list(param.shape)))
                raise RuntimeError
        else:
            print('Saved parameter named [%s] is skipped' % name)
    mark = True
    for name in is_loaded:
        if not is_loaded[name]:
            print('Parameter named [%s] is randomly initialized' % name)
            mark = False
    if mark:
        print('All parameters are initialized using [%s]' % path)


# 加载预训练的模型参数
load_networks(unet, './checkpoint/UNET_model_MY.pth')
unet.eval()  # 设置模型为评估模式，不进行梯度计算

## 单张牙齿X光图像预处理
# 分割单张牙齿X光图像
img = cv2.imread('./dataset/test/xray/105.png')[..., ::-1]  # 读取单张牙齿X光图像
img = np.float32(img) / 255.  # 将图像转换为浮点数并归一化
img = np.transpose(img, (2, 0, 1))  # 转置图像数组维度
r, g, b = img[0:1, :, :], img[1:2, :, :], img[2:3, :, :]  # 分离图像的RGB通道
img = 0.2989 * r + 0.5870 * g + 0.1140 * b  # 灰度化处理
print(img.shape)
img = torch.from_numpy(img)  # 将图像转换为PyTorch张量
img = img[None, ...]  # 添加一个额外的维度
# 查看分割结果
mask = unet(img)  # 使用U_Net模型预测图像掩模
mask = torch.clamp(mask[0].detach() * 255, 0, 255).round()  # 处理模型输出并四舍五入
mask = np.array(mask)  # 转换为NumPy数组
mask = np.uint8(mask[0])  # 转换为8位无符号整数类型
plt.imshow(mask)  # 显示分割结果
plt.show()

# 查看预训练分类模型在测试数据集上的准确率
# 计算IOU和准确率
def calc_iou(target, prediction):
    target = np.uint8(np.array(target / 255.).flatten() > 0.5)  # 处理目标掩模
    prediction = np.uint8(np.array(prediction / 255.).flatten() > 0.5)  # 处理预测掩模
    TP = (prediction * target).sum()  # 计算真正例
    FN = ((1 - prediction) * target).sum()  # 计算假负例
    TN = ((1 - prediction) * (1 - target)).sum()  # 计算真负例
    FP = (prediction * (1 - target)).sum()  # 计算假正例

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)  # 计算准确率
    iou = TP / (TP + FP + FN + 1e-4)  # 计算IOU
    return iou, acc


# 定义用于处理X射线图像分割数据的自定义PyTorch数据集
class XRayDataset(Dataset):

    # 初始化
    def __init__(self, images_path_list, labels_path_list, split='Train', augmentation=True, device='cuda:1',
                 image_size=(512, 512)):
        self.images = images_path_list
        self.labels = labels_path_list
        self.augmentation = augmentation
        self.device = device
        self.split = split

        self.transform = transforms.Compose([
            transforms.Grayscale(),  # 将图像转换为灰度
            transforms.ToTensor()  # 将图像转换为张量
        ])

        if self.augmentation:
            self.same_augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转
            ])

        # 加载图像和标签
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
        idx = idx % len(self.names)
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        # 如果启用了数据增强
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


## 构建测试数据DataLoader
# 设置图像和标签数据的路径
image_path_test = './dataset/test/xray/'
labels_path_test = './dataset/test/mask/'

# 获取图像目录中的所有文件名
image_names_test = [filename for filename in os.listdir(image_path_test)]
label_names_test = [filename for filename in os.listdir(labels_path_test)]

# 创建完整的测试图像和标签文件路径
test_image_path = [image_path_test + file_name for file_name in image_names_test]
test_mask_path = [labels_path_test + file_name for file_name in label_names_test]

# 创建测试数据集
test_dataset = XRayDataset(
    images_path_list=test_image_path,
    labels_path_list=test_mask_path,
    augmentation=False,
    split='Test',
    device=torch.device('cpu'),
)

toPIL = transforms.ToPILImage()

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
print('test_dataset：', len(test_dataloader))

# 对整个测试数据集进行测试并保存结果
if not os.path.exists('./output'):
    os.mkdir('./output')
tqdm_test = tqdm(test_dataloader)
iou = [0.0] * len(test_dataloader)
acc = [0.0] * len(test_dataloader)
unet.eval()
for i, data in enumerate(tqdm_test):
    with torch.no_grad():
        mask = unet(data['rgb'])
    mask = torch.clamp(mask[0].detach() * 255, 0, 255).round()
    mask = np.array(mask)
    mask = np.uint8(mask[0])
    gt = torch.clamp(data['label'][0].detach() * 255, 0, 255).round()
    iou[i], acc[i] = calc_iou(gt, mask)
    cv2.imwrite(os.path.join('./output/', data['fname'][0]), mask)
print('IOU：', np.mean(iou))
print('Mask Accuracy：', np.mean(acc))

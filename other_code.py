import torch
import cv2  # 导入OpenCV库，用于图像处理
from matplotlib import pyplot as plt
from network import U_Net  # 导入自定义的U_Net网络模型

# 查看pytorch版本
print(torch.__version__)
print('\n')

# 查看测试图片和对应Mask
# 查看测试图片
img = cv2.imread('./dataset/test/xray/101.png')
plt.imshow(img)
plt.show()
# 查看测试图片对应Mask
img = cv2.imread('./dataset/test/mask/101.png')
plt.imshow(img)
plt.show()

# 构建和查看分割网络结构
unet = U_Net()
print(unet)
print('\n')

# 查看参数量大小
print('---------- Size of Parameters  -------------')
num_params = 0
for param in unet.parameters():
    num_params += param.numel()
print('[Network %s] Total number of parameters : %.3f M'
      % ('UNet', num_params / 1e6))
print('-----------------------------------------------')

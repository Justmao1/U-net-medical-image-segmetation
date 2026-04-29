import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import XRayDataset
from models import UNet, AttentionUNet, UNetPP
from utils import calc_iou, load_networks


MODEL_REGISTRY = {
    'unet': UNet,
    'attention_unet': AttentionUNet,
    'unet_pp': UNetPP,
}


def get_args():
    parser = argparse.ArgumentParser(description='U-Net 医学图像分割测试')
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')
    parser.add_argument('--model', type=str, default='unet',
                        choices=MODEL_REGISTRY.keys(), help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径')
    parser.add_argument('--image_dir', type=str, default=None, help='测试图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='测试标签目录')
    parser.add_argument('--output_dir', type=str, default=None, help='预测结果保存目录')
    return parser.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = get_args()

    # 加载配置
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    model_name = args.model or cfg.get('model', 'unet')
    in_ch = cfg.get('in_channels', 1)
    out_ch = cfg.get('out_channels', 1)

    test_cfg = cfg.get('test', {})
    checkpoint_path = args.checkpoint or test_cfg.get('checkpoint', './checkpoint/best_model.pth')
    image_dir = args.image_dir or test_cfg.get('image_dir', './dataset/test/xray')
    mask_dir = args.mask_dir or test_cfg.get('mask_dir', './dataset/test/mask')
    output_dir = args.output_dir or test_cfg.get('output_dir', './output')

    # 构建模型并加载权重
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_ch=in_ch, out_ch=out_ch)
    load_networks(model, checkpoint_path)
    model.eval()

    # 加载测试数据
    image_names = sorted(os.listdir(image_dir))
    label_names = sorted(os.listdir(mask_dir))

    test_image_paths = [os.path.join(image_dir, f) for f in image_names]
    test_mask_paths = [os.path.join(mask_dir, f) for f in label_names]

    test_dataset = XRayDataset(
        images_path_list=test_image_paths,
        labels_path_list=test_mask_paths,
        split='Test',
        augmentation=False,
        device='cpu',
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Test dataset: {len(test_dataloader)} images')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 推理并计算指标
    iou_list = []
    acc_list = []

    for data in tqdm(test_dataloader, desc='Testing'):
        with torch.no_grad():
            mask = model(data['rgb'])

        # 将 logits 转为 0-255 掩码
        mask = torch.sigmoid(mask)
        mask = torch.clamp(mask[0] * 255, 0, 255).round()
        mask = np.uint8(np.array(mask)[0])

        gt = torch.clamp(data['label'][0] * 255, 0, 255).round()
        iou, acc = calc_iou(gt, mask)
        iou_list.append(iou)
        acc_list.append(acc)

        cv2.imwrite(os.path.join(output_dir, data['fname'][0]), mask)

    print(f'Mean IoU:       {np.mean(iou_list):.4f}')
    print(f'Mean Accuracy:  {np.mean(acc_list):.4f}')


if __name__ == '__main__':
    main()

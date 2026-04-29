import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import XRayDataset
from models import UNet, AttentionUNet, UNetPP
from tqdm import tqdm


MODEL_REGISTRY = {
    'unet': UNet,
    'attention_unet': AttentionUNet,
    'unet_pp': UNetPP,
}


def get_args():
    parser = argparse.ArgumentParser(description='U-Net 医学图像分割训练')
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')
    parser.add_argument('--model', type=str, default='unet',
                        choices=MODEL_REGISTRY.keys(), help='模型类型')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--image_dir', type=str, default=None, help='训练图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='训练标签目录')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint 保存目录')
    parser.add_argument('--augmentation', action='store_true', help='启用数据增强')
    return parser.parse_args()


def load_config(config_path):
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = get_args()

    # 加载配置文件（如果指定）
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # 命令行参数优先于配置文件
    model_name = args.model or cfg.get('model', 'unet')
    in_ch = cfg.get('in_channels', 1)
    out_ch = cfg.get('out_channels', 1)

    train_cfg = cfg.get('train', {})
    epochs = args.epochs or train_cfg.get('epochs', 50)
    batch_size = args.batch_size or train_cfg.get('batch_size', 2)
    lr = args.lr or train_cfg.get('lr', 0.001)
    image_dir = args.image_dir or train_cfg.get('image_dir', './dataset/train/xray')
    mask_dir = args.mask_dir or train_cfg.get('mask_dir', './dataset/train/mask')
    checkpoint_dir = args.checkpoint_dir or train_cfg.get('checkpoint_dir', './checkpoint')
    augmentation = args.augmentation or train_cfg.get('augmentation', False)
    save_every = train_cfg.get('save_every', 10)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    image_names = sorted(os.listdir(image_dir))
    label_names = sorted(os.listdir(mask_dir))

    train_image_paths = [os.path.join(image_dir, f) for f in image_names]
    train_mask_paths = [os.path.join(mask_dir, f) for f in label_names]

    train_dataset = XRayDataset(
        images_path_list=train_image_paths,
        labels_path_list=train_mask_paths,
        split='Train',
        augmentation=augmentation,
        device=device,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_ch=in_ch, out_ch=out_ch).to(device)
    print(f'Model: {model_name}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 创建 checkpoint 目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch}/{epochs}]')
        for data in pbar:
            inputs = data['rgb'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = running_loss / len(train_dataloader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}')

        # 保存 best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({'state_dict': model.state_dict()}, save_path)
            print(f'  -> Saved best model (loss={best_loss:.4f})')

        # 定期保存 checkpoint
        if epoch % save_every == 0:
            save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save({'state_dict': model.state_dict()}, save_path)

    print('Training complete.')


if __name__ == '__main__':
    main()

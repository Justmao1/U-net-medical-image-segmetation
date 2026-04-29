# U-Net Medical Image Segmentation

Medical image segmentation project for dental X-ray images, supporting three U-Net variants.

## Supported Models

| Model | Description |
|-------|-------------|
| `unet` | Standard U-Net |
| `attention_unet` | Attention U-Net with attention-gated skip connections |
| `unet_pp` | U-Net++ with dense nested skip connections |

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
dataset/
├── train/
│   ├── xray/       # Training images
│   └── mask/       # Corresponding binary segmentation masks
└── test/
    ├── xray/       # Test images
    └── mask/       # Corresponding binary segmentation masks
```

## Training

```bash
# Default configuration
python train.py --model unet

# With YAML config file
python train.py --config configs/default.yaml

# Override parameters via command line
python train.py --model attention_unet --epochs 100 --batch_size 4 --lr 0.0005
```

## Testing

```bash
python test.py --model unet --checkpoint ./checkpoint/best_model.pth
```

## Project Structure

```
├── models/              # Model definitions
│   ├── unet.py          # Standard U-Net
│   ├── attention_unet.py # Attention U-Net
│   └── unet_pp.py       # U-Net++
├── datasets/            # Dataset classes
│   └── xray_dataset.py  # X-ray image dataset
├── utils/               # Utility functions
│   ├── metrics.py       # IoU / Accuracy calculation
│   └── checkpoint.py    # Model weight loading
├── configs/             # Configuration files
│   └── default.yaml     # Default config
├── train.py             # Training entry point
├── test.py              # Testing entry point
└── requirements.txt     # Dependencies
```

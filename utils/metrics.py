import numpy as np


def calc_iou(target, prediction):
    """计算 IoU 和 Accuracy

    Args:
        target: 真实掩码（numpy array，0-255 范围）
        prediction: 预测掩码（numpy array，0-255 范围）

    Returns:
        (iou, accuracy)
    """
    target = np.uint8(np.array(target / 255.0).flatten() > 0.5)
    prediction = np.uint8(np.array(prediction / 255.0).flatten() > 0.5)

    TP = (prediction * target).sum()
    FN = ((1 - prediction) * target).sum()
    TN = ((1 - prediction) * (1 - target)).sum()
    FP = (prediction * (1 - target)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou = TP / (TP + FP + FN + 1e-4)
    return iou, acc

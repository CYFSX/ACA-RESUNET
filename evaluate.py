import torch
from torch.utils.data import DataLoader
from RA_UNet import UNetResNet50
from dataset import SegmentationDataset
from utils import MetricsCalculator, CLASS_NAMES
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置全局中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_test_transform():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def evaluate_model(model, data_loader, device):
    model.eval()
    metrics_calculator = MetricsCalculator()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc='Evaluating'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            metrics_calculator.update(predictions, masks)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    metrics = metrics_calculator.compute()
    cm = confusion_matrix(all_targets, all_predictions)
    return metrics, cm

def plot_metrics(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 保存详细指标到CSV
    metrics_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Dice Coefficient': metrics['class_dice'],
        'IoU': metrics['class_iou'],
        'Accuracy': metrics['class_accuracy']
    })
    metrics_df.to_csv(os.path.join(save_dir, 'detailed_metrics.csv'), index=False, encoding='utf-8-sig')

    overall_metrics = pd.DataFrame({
        'Metric': ['Mean Dice Coefficient', 'Mean IoU', 'Accuracy'],
        'Value': [metrics['mean_dice'], metrics['mean_iou'], metrics['accuracy']]
    })
    overall_metrics.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False, encoding='utf-8-sig')

    # 绘制总体指标
    plot_bar_chart(overall_metrics, os.path.join(save_dir, 'overall_metrics.png'))

    # 绘制每个类别的指标
    plot_per_class_metrics(metrics, save_dir)

def plot_bar_chart(data, save_path):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['Metric'], data['Value'])
    plt.title('Overall Evaluation Metrics')
    plt.ylabel('Metric Value')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(metrics, save_dir):
    plots = [
        ('Dice Coefficient', 'dice_per_class.png', metrics['class_dice']),
        ('IoU', 'iou_per_class.png', metrics['class_iou']),
        ('Accuracy', 'accuracy_per_class.png', metrics['class_accuracy'])
    ]

    for title, filename, values in plots:
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(values)), values)
        plt.title(f'Per-Class {title}')
        plt.xlabel('Class')
        plt.ylabel(title)
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_confusion_matrix(cm, save_dir):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    model = UNetResNet50(n_channels=3, n_classes=16)
    checkpoint = torch.load('experiments/experiment_20250327_205138/checkpoints/resnet50_unet.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logging.info("Model loaded")

    test_dataset = SegmentationDataset(img_dir='data/images', label_dir='data/labels', transform=get_test_transform(), preload=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    logging.info(f"Test set size: {len(test_dataset)}")

    metrics, confusion_mat = evaluate_model(model, test_loader, device)
    logging.info(f"Evaluation results: Mean Dice={metrics['mean_dice']:.4f}, Mean IoU={metrics['mean_iou']:.4f}, Accuracy={metrics['accuracy']:.4f}")

    save_dir = 'evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    plot_metrics(metrics, save_dir)
    plot_confusion_matrix(confusion_mat, save_dir)
    logging.info(f"Results saved to {save_dir}")

if __name__ == '__main__':
    main()
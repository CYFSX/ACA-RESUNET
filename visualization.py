import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_training_progress(exp_dir, metrics_history, step):
    """绘制损失、Dice、IoU 曲线"""
    fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    if metrics_history.get('train_loss'):
        axs[0].plot(metrics_history['train_loss'], label='Train')
    if metrics_history.get('val_loss'):
        axs[0].plot(metrics_history['val_loss'], label='Val')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Steps')
    axs[0].legend()

    # Dice
    if metrics_history.get('train_dice'):
        axs[1].plot(metrics_history['train_dice'], label='Train')
    if metrics_history.get('val_dice'):
        axs[1].plot(metrics_history['val_dice'], label='Val')
    axs[1].set_title('Dice Coefficient')
    axs[1].set_xlabel('Steps')
    axs[1].legend()

    # IoU
    if metrics_history.get('train_iou'):
        axs[2].plot(metrics_history['train_iou'], label='Train')
    if metrics_history.get('val_iou'):
        axs[2].plot(metrics_history['val_iou'], label='Val')
    axs[2].set_title('IoU')
    axs[2].set_xlabel('Steps')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'metrics_step_{step}.png'))
    plt.close()


def plot_lr_schedule(exp_dir, lr_history, step):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history)
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('LR')
    plt.savefig(os.path.join(exp_dir, f'lr_schedule_{step}.png'))
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """保存混淆矩阵热力图"""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
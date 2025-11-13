import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
import pandas as pd

from constants import CLASS_NAMES_EN as CLASS_NAMES


class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        n = len(CLASS_NAMES)
        self.intersection = torch.zeros(n)
        self.union = torch.zeros(n)
        self.target_sum = torch.zeros(n)
        self.pred_sum = torch.zeros(n)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        device = pred.device
        if self.intersection.device != device:
            self.intersection = self.intersection.to(device)
            self.union = self.union.to(device)
            self.target_sum = self.target_sum.to(device)
            self.pred_sum = self.pred_sum.to(device)

        pred = pred.view(-1)
        target = target.view(-1)

        for i in range(len(CLASS_NAMES)):
            p_mask = pred == i
            t_mask = target == i
            self.intersection[i] += (p_mask & t_mask).sum().float()
            self.union[i] += (p_mask | t_mask).sum().float()
            self.target_sum[i] += t_mask.sum().float()
            self.pred_sum[i] += p_mask.sum().float()

        self.correct_pixels += (pred == target).sum().float()
        self.total_pixels += target.numel()

    def compute(self):
        eps = 1e-7
        dice = 2 * self.intersection / (self.pred_sum + self.target_sum + eps)
        iou = self.intersection / (self.union + eps)
        acc = self.correct_pixels / (self.total_pixels + eps)
        class_acc = (self.intersection / (self.target_sum + eps)).tolist()

        # MPA: exclude background (class 0) if exists
        mpa = np.mean(class_acc[1:]) if len(class_acc) > 1 else np.mean(class_acc)

        prec = self.intersection / (self.pred_sum + eps)
        rec = self.intersection / (self.target_sum + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        tn = self.total_pixels - self.pred_sum.sum() - self.target_sum.sum() + self.intersection.sum()
        fp = self.pred_sum.sum() - self.intersection.sum()
        spec = tn / (tn + fp + eps)

        return {
            'mean_dice': dice.mean().item(),
            'mean_iou': iou.mean().item(),
            'mpa': mpa,
            'accuracy': acc.item(),
            'mean_f1': f1.mean().item(),
            'mean_sensitivity': rec.mean().item(),
            'specificity': spec.item(),
            'class_dice': dice.tolist(),
            'class_iou': iou.tolist(),
            'class_accuracy': class_acc,
            'class_f1': f1.tolist(),
            'class_precision': prec.tolist(),
            'class_recall': rec.tolist()
        }


class Visualizer:
    def __init__(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
        np.random.seed(42)
        self.color_map = np.random.randint(0, 255, (len(CLASS_NAMES), 3), dtype=np.uint8)
        self.color_map[0] = [0, 0, 0]  # background black

    def plot_training_curves(self, train_losses, val_metrics, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(train_losses, label='Train Loss', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for name, values in val_metrics.items():
            if name in ['mean_dice', 'mean_iou', 'accuracy']:
                label = {'mean_dice': 'Dice', 'mean_iou': 'IoU', 'accuracy': 'Acc'}[name]
                ax2.plot(values, label=label, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_prediction(self, image, mask, pred, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        mask_rgb = self.color_map[mask.cpu().numpy()]
        pred_rgb = self.color_map[pred.cpu().numpy()]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_np)
        axs[1].imshow(mask_rgb)
        axs[2].imshow(pred_rgb)
        for ax, title in zip(axs, ['Image', 'Ground Truth', 'Prediction']):
            ax.set_title(title)
            ax.axis('off')

        handles = [plt.Rectangle((0,0),1,1, color=self.color_map[i]/255) for i in range(len(CLASS_NAMES))]
        fig.legend(handles, CLASS_NAMES, loc='center right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, value):
        if self.best is None or \
           (self.mode == 'max' and value > self.best + self.min_delta) or \
           (self.mode == 'min' and value < self.best - self.min_delta):
            self.best = value
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop


def print_metrics(metrics, phase='val'):
    print(f"\n{phase.upper()} Results:")
    print(f"mIoU: {metrics['mean_iou']:.4f} | Dice: {metrics['mean_dice']:.4f} | "
          f"MPA: {metrics['mpa']:.4f} | Acc: {metrics['accuracy']:.4f}")
    print("Per-class IoU:", [f"{v:.3f}" for v in metrics['class_iou']])


def save_results(results_dir, best_metrics, history, model_info, total_time):
    os.makedirs(results_dir, exist_ok=True)

    main = {
        'mIoU': best_metrics['mean_iou'],
        'MPA': best_metrics['mpa'],
        'Dice': best_metrics['mean_dice'],
        'Accuracy': best_metrics['accuracy'],
        'F1': best_metrics['mean_f1'],
        'Sensitivity': best_metrics['mean_sensitivity'],
        'Specificity': best_metrics['specificity']
    }

    pd.DataFrame([main]).to_csv(os.path.join(results_dir, 'main_metrics.csv'), index=False)

    class_df = pd.DataFrame([
        {
            'Class': CLASS_NAMES[i],
            'Dice': best_metrics['class_dice'][i],
            'IoU': best_metrics['class_iou'][i],
            'Acc': best_metrics['class_accuracy'][i]
        }
        for i in range(len(CLASS_NAMES))
    ])
    class_df.to_csv(os.path.join(results_dir, 'class_metrics.csv'), index=False)

    summary = {
        'total_time_h': total_time / 3600,
        'epochs': len(history),
        'best_epoch': max(range(len(history)), key=lambda i: history[i].get('val_mean_dice', 0)) + 1
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
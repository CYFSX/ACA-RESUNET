import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import torch.nn as nn
from dataset import SegmentationDataset
from RA_UNet import UNetResNet50
import logging
from utils import MetricsCalculator, Visualizer, CLASS_NAMES, EarlyStopping, PaperResultsSaver
from config import config
from losses import get_loss_function
import json
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 固定随机种子
def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 数据预加载器（简化版）
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_images, self.next_masks = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_images = self.next_images.to(self.device, non_blocking=True)
                self.next_masks = self.next_masks.to(self.device, dtype=torch.long, non_blocking=True)
        except StopIteration:
            self.next_images = None
            self.next_masks = None

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_images is None:
            raise StopIteration
        images, masks = self.next_images, self.next_masks
        self.preload()
        return images, masks

    def __iter__(self):
        return self

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    metrics = MetricsCalculator()
    prefetcher = DataPrefetcher(loader, device)
    pbar = tqdm(range(len(loader)), desc='Train', leave=False)

    for _ in pbar:
        images, masks = next(prefetcher)
        with autocast():
            out = model(images)
            main_out = out[0] if isinstance(out, tuple) else out
            aux_out = out[1:] if isinstance(out, tuple) and len(out) > 1 else None
            loss_res = criterion(main_out, masks, aux_out)
            loss = loss_res[0] if isinstance(loss_res, tuple) else loss_res

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIPPING)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            pred = main_out.argmax(1)
            metrics.update(pred, masks)

        total_loss += loss.item()
        current = metrics.compute()
        pbar.set_postfix(loss=loss.item(), dice=current['mean_dice'], iou=current['mean_iou'])

        if _ % config.MEMORY_CLEANUP_FREQ == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    final_metrics = metrics.compute()
    final_metrics['loss'] = avg_loss
    return final_metrics

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = MetricsCalculator()
    pbar = tqdm(loader, desc='Val', leave=False)

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)

            with autocast():
                out = model(images)
                main_out = out[0] if isinstance(out, tuple) else out
                aux_out = out[1:] if isinstance(out, tuple) and len(out) > 1 else None
                loss_res = criterion(main_out, masks, aux_out)
                loss = loss_res[0] if isinstance(loss_res, tuple) else loss_res

            pred = main_out.argmax(1)
            metrics.update(pred, masks)
            total_loss += loss.item()

            current = metrics.compute()
            pbar.set_postfix(loss=loss.item(), dice=current['mean_dice'], iou=current['mean_iou'])

            if len(metrics.history) % (config.MEMORY_CLEANUP_FREQ // 2) == 0:
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    final_metrics = metrics.compute()
    final_metrics['loss'] = avg_loss
    return final_metrics

class ExperimentManager:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = f"experiments/experiment_{self.timestamp}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/logs", exist_ok=True)
        self.visualizer = Visualizer()
        self.train_log = []
        self.val_log = []
        self.best_dice = 0

        cfg = {
            'model': 'UNetResNet50',
            'lr': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.NUM_EPOCHS,
            'classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        }
        with open(f"{self.exp_dir}/config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

    def log(self, epoch, train_m, val_m):
        self.train_log.append(train_m)
        self.val_log.append(val_m)
        with open(f"{self.exp_dir}/logs/metrics_{epoch}.json", 'w') as f:
            json.dump({'epoch': epoch, 'train': train_m, 'val': val_m}, f, indent=2)

        # 绘图
        train_loss = [x['loss'] for x in self.train_log]
        val_dice = [x['mean_dice'] for x in self.val_log]
        self.visualizer.plot_training_curves(train_loss, {'mean_dice': val_dice},
                                             save_path=f"{self.exp_dir}/logs/curves.png")

    def save_best(self, state, is_best):
        if is_best:
            path = f"{self.exp_dir}/checkpoints/best_model.pth"
            torch.save(state, path)

    def visualize(self, model, loader, device, epoch):
        if not config.VISUALIZATION_ENABLED or epoch % config.VIS_FREQ != 0:
            return
        model.eval()
        try:
            images, masks = next(iter(loader))
            images = images[:config.MAX_VIS_SAMPLES].to(device)
            with torch.no_grad():
                out = model(images)
                preds = (out[0] if isinstance(out, tuple) else out).argmax(1)
            for i in range(len(images)):
                self.visualizer.visualize_prediction(
                    images[i], masks[i], preds[i],
                    save_path=f"{self.exp_dir}/logs/epoch_{epoch}_sample_{i}.png"
                )
        except Exception as e:
            logging.warning(f"Vis failed: {e}")

def main():
    set_seed()
    device = config.DEVICE
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据集
    full_set = SegmentationDataset(config.IMG_DIR, config.LABEL_DIR, preload=False)
    n = len(full_set)
    n_train = int(config.TRAIN_RATIO * n)
    train_set, val_set = random_split(full_set, [n_train, n - n_train],
                                      generator=torch.Generator().manual_seed(config.SEED))

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=config.PREFETCH_FACTOR)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=config.PREFETCH_FACTOR)

    # 模型
    model = UNetResNet50(n_channels=config.N_CHANNELS, n_classes=config.N_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 损失与优化
    class_weights = full_set.get_class_weights().to(device)
    criterion = get_loss_function(
        num_classes=config.N_CLASSES,
        loss_weights=[0.3, 0.4, 0.2, 0.1],
        label_smoothing=config.LABEL_SMOOTHING,
        class_weights=class_weights,
        use_lovasz=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.COSINE_T0,
                                            T_mult=config.COSINE_T_MULT,
                                            eta_min=config.COSINE_ETA_MIN)
    scaler = GradScaler()
    exp = ExperimentManager()
    early_stop = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')
    history = []
    best_metrics = None
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_m = validate_epoch(model, val_loader, criterion, device)

        is_best = val_m['mean_dice'] > exp.best_dice
        if is_best:
            exp.best_dice = val_m['mean_dice']
            best_metrics = val_m.copy()

        if (epoch + 1) % config.SAVE_FREQ == 0:
            exp.log(epoch, train_m, val_m)
        if (epoch + 1) % config.VIS_FREQ == 0:
            exp.visualize(model, val_loader, device, epoch)

        exp.save_best({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': exp.best_dice
        }, is_best)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
                     f"Train Loss: {train_m['loss']:.4f} Dice: {train_m['mean_dice']:.4f} | "
                     f"Val Loss: {val_m['loss']:.4f} Dice: {val_m['mean_dice']:.4f} | "
                     f"Time: {elapsed:.1f}s ({total_elapsed/60:.1f} min)")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_m['loss'],
            'val_loss': val_m['loss'],
            'val_dice': val_m['mean_dice'],
            'val_iou': val_m['mean_iou'],
            'val_mpa': val_m.get('mpa', 0)
        })

        if early_stop(val_m['mean_dice']):
            logging.info("Early stopping")
            break

        torch.cuda.empty_cache()

    try:
        total_params = sum(p.numel() for p in model.parameters())
        paper_saver = PaperResultsSaver(exp.exp_dir)
        paper_saver.save_final_results(
            best_metrics=best_metrics or history[-1],
            training_history=history,
            model_info={'total_params': total_params},
            training_time=time.time() - start_time
        )
        logging.info("Paper results saved")
    except Exception as e:
        logging.error(f"Failed to save paper results: {e}")

    logging.info(f"Training finished. Best Dice: {exp.best_dice:.4f}")
    logging.info(f"Results in: {exp.exp_dir}")

if __name__ == '__main__':
    main()
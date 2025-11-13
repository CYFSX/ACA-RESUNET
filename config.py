"""
配置文件：管理所有超参数和设置
"""
import torch

class Config:
    # 数据路径
    IMG_DIR = 'data/images'
    LABEL_DIR = 'data/labels'

    # 图像与数据加载参数
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4

    # 模型参数
    N_CHANNELS = 3
    N_CLASSES = 16
    PRETRAINED = True

    # 训练参数
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    SEED = 42

    # 损失函数相关
    LOSS_WEIGHTS = [0.5, 0.5]      # CE 和 Dice 的权重
    AUX_WEIGHTS = [0.4, 0.3, 0.3]  # 辅助输出的损失权重
    LABEL_SMOOTHING = 0.1

    # 学习率调度
    SCHEDULER_TYPE = 'cosine'      # 可选: 'cosine', 'plateau', 'step'
    COSINE_T0 = 20
    COSINE_T_MULT = 2
    COSINE_ETA_MIN = 1e-6

    # 早停策略
    EARLY_STOP_PATIENCE = 30
    EARLY_STOP_MIN_DELTA = 1e-5
    EARLY_STOP_MODE = 'max'

    # 数据增强概率
    HORIZONTAL_FLIP_P = 0.5
    BRIGHTNESS_CONTRAST_P = 0.2
    SHIFT_SCALE_ROTATE_P = 0.2
    NOISE_P = 0.2

    # 训练优化选项
    MEMORY_CLEANUP_FREQ = 5
    GRADIENT_CLIPPING = 1.0
    USE_AMP = True
    GRADIENT_ACCUMULATION_STEPS = 1

    # 保存与可视化
    SAVE_FREQ = 5
    VIS_FREQ = 5
    VISUALIZATION_ENABLED = True
    MAX_VIS_SAMPLES = 4
    VIS_DPI = 300
    VIS_FIGSIZE = (15, 5)

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集划分比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    @classmethod
    def to_dict(cls):
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

    @classmethod
    def update_from_dict(cls, config_dict):
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"警告: 配置项 {key} 不存在")


# 实例化默认配置
config = Config()
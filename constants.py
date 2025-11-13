CLASSES = [
    'background',
    'book',
    'circular spirit level',
    'cord',
    'disk',
    'fixed pulley',
    'hand',
    'paper',
    'pen',
    'prism',
    'ring',
    'rotational inertia host',
    'rotational inertia tester',
    'spectrometer',
    'transceiver',
    'weight'
]

CLASS_NAMES_EN = [
    'Background', 'Book', 'Spirit Level', 'Cord', 'Disk',
    'Fixed Pulley', 'Hand', 'Paper', 'Pen', 'Prism',
    'Ring', 'Rotational Inertia Host', 'Rotational Inertia Tester',
    'Spectrometer', 'Transceiver', 'Weight'
]

CLASS_NAMES_ZH = [
    '背景', '书本', '水准器', '绳子', '圆盘',
    '定滑轮', '手', '纸', '笔', '棱镜',
    '光环', '转动惯量主机', '转动惯量测试仪', '光谱仪', '收发器', '砝码'
]

LABEL_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

DEFAULT_N_CLASSES = len(CLASSES)
DEFAULT_N_CHANNELS = 3
DEFAULT_IMG_SIZE = 256

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 200

DEFAULT_LOSS_WEIGHTS = [0.3, 0.4, 0.2, 0.1]
DEFAULT_AUX_WEIGHTS = [0.4, 0.3, 0.3]

DEFAULT_AUGMENTATION_PARAMS = {
    'horizontal_flip_p': 0.5,
    'brightness_contrast_p': 0.2,
    'shift_scale_rotate_p': 0.3,
    'noise_p': 0.3
}

RTX_3080_TI_CONFIG = {
    'batch_size': 16,
    'num_workers': 8,
    'pin_memory': True,
    'prefetch_factor': 4,
    'memory_cleanup_freq': 5,
    'use_amp': True
}
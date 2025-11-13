import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PredictionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if not self.img_files:
            raise RuntimeError(f"No image files found in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.img_files[idx]


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, preload=False, mask_png_dir='data/masks_png'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_png_dir = mask_png_dir
        self.preload = preload

        from constants import CLASSES, LABEL_MAP
        self.classes = CLASSES
        self.class_to_idx = LABEL_MAP

        self.valid_pairs = []
        self._validate_pairs()
        if not self.valid_pairs:
            raise RuntimeError(f"No valid image-label pairs found in {img_dir}")

        logging.info(f"Found {len(self.valid_pairs)} valid image-label pairs")

        self.transform = transform if transform else self._get_default_transform()
        self.cached_data = []
        if self.preload:
            self._preload_data()

    def _get_default_transform(self):
        return A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2.0),
                A.ColorJitter(),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _validate_pairs(self):
        for img_name in os.listdir(self.img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            json_name = os.path.splitext(img_name)[0] + '.json'
            json_path = os.path.join(self.label_dir, json_name)
            if not os.path.exists(json_path):
                continue
            img_path = os.path.join(self.img_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                self.valid_pairs.append((img_name, json_name))
            except Exception:
                pass

    def _preload_data(self):
        for img_name, json_name in tqdm(self.valid_pairs):
            try:
                img_path = os.path.join(self.img_dir, img_name)
                json_path = os.path.join(self.label_dir, json_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                mask = self._create_mask(json_path, image.shape[:2])
                self.cached_data.append({'image_path': img_path, 'mask': mask})
            except Exception as e:
                logging.error(f"Preload error for {img_path}: {e}")

    def _create_mask(self, json_path, img_size):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"JSON load error: {e}")
            return np.zeros(img_size, dtype=np.uint8)

        mask = np.zeros(img_size, dtype=np.uint8)
        for shape in data.get('shapes', []):
            label = shape.get('label', '').strip()
            class_idx = self.class_to_idx.get(label)
            if class_idx is None:
                continue
            points = np.array(shape['points'], dtype=np.int32)
            if len(points) < 3:
                continue
            if shape.get('shape_type') == 'rectangle':
                cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), class_idx, -1)
            else:
                cv2.fillPoly(mask, [points], class_idx)
        return mask

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        if self.preload:
            data = self.cached_data[idx]
            image = cv2.imread(data['image_path'])
            mask = data['mask']
        else:
            img_name, json_name = self.valid_pairs[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                raise RuntimeError(f"Cannot read image: {img_path}")

            mask_png_path = os.path.join(self.mask_png_dir, os.path.splitext(img_name)[0] + '.png')
            if os.path.exists(mask_png_path):
                mask = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Cannot read mask: {mask_png_path}")
            else:
                json_path = os.path.join(self.label_dir, json_name)
                mask = self._create_mask(json_path, image.shape[:2])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        return image, mask

    @torch.no_grad()
    def get_class_weights(self):
        class_counts = torch.zeros(len(self.classes))
        if self.preload:
            for data in self.cached_data:
                mask = data['mask']
                for i in range(len(self.classes)):
                    class_counts[i] += (mask == i).sum()
        else:
            for img_name, json_name in self.valid_pairs:
                img_path = os.path.join(self.img_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                json_path = os.path.join(self.label_dir, json_name)
                mask = self._create_mask(json_path, image.shape[:2])
                for i in range(len(self.classes)):
                    class_counts[i] += (mask == i).sum()

        class_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
        weights = 1.0 / torch.log(1.2 + class_counts)
        weights = weights / weights.sum() * len(self.classes)
        return weights


if __name__ == '__main__':
    try:
        dataset = SegmentationDataset('data/images', 'data/labels', preload=False)
        logging.info(f"Dataset size: {len(dataset)}")
        image, mask = dataset[0]
        logging.info(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        logging.info(f"Unique classes: {torch.unique(mask)}")
        weights = dataset.get_class_weights()
        for cls, w in zip(dataset.classes, weights):
            logging.info(f"{cls}: {w:.4f}")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise
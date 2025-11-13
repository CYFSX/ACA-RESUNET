import os
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm
from utils import LABEL_MAP


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_mask(json_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return mask

    for shape in data.get('shapes', []):
        label = shape.get('label', '').strip()
        cls_id = LABEL_MAP.get(label)
        if cls_id is None:
            continue
        pts = np.array(shape.get('points', []), dtype=np.int32)
        if len(pts) < 2:
            continue
        stype = shape.get('shape_type', 'polygon')
        if stype == 'polygon':
            cv2.fillPoly(mask, [pts], int(cls_id))
        elif stype == 'rectangle':
            cv2.rectangle(mask, tuple(pts[0]), tuple(pts[1]), int(cls_id), -1)
    return mask


def make_overlay(img, mask, alpha=0.4):
    rng = np.random.default_rng(42)
    color_map = rng.integers(0, 255, size=(max(1, mask.max() + 1), 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]
    colored_mask = color_map[mask]
    return cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='data/images')
    parser.add_argument('--json_dir', default='data/labels')
    parser.add_argument('--out_mask_dir', default='data/masks_png')
    parser.add_argument('--overlay_dir', default='')
    parser.add_argument('--alpha', type=float, default=0.4)
    args = parser.parse_args()

    ensure_dir(args.out_mask_dir)
    if args.overlay_dir:
        ensure_dir(args.overlay_dir)

    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"图像目录不存在: {args.img_dir}")
    if not os.path.isdir(args.json_dir):
        raise FileNotFoundError(f"标注目录不存在: {args.json_dir}")

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(img_exts)]

    for name in tqdm(img_files, desc='Processing'):
        img_path = os.path.join(args.img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        json_path = os.path.join(args.json_dir, os.path.splitext(name)[0] + '.json')
        if not os.path.exists(json_path):
            continue

        mask = create_mask(json_path, h, w)
        mask_path = os.path.join(args.out_mask_dir, os.path.splitext(name)[0] + '.png')
        cv2.imwrite(mask_path, mask)

        if args.overlay_dir:
            overlay = make_overlay(img, mask, args.alpha)
            overlay_path = os.path.join(args.overlay_dir, os.path.splitext(name)[0] + '.png')
            cv2.imwrite(overlay_path, overlay)

    print('Done.')


if __name__ == '__main__':
    main()
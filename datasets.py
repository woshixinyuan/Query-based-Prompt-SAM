# datasets.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random


def read_split_files(file_path):
    """Read a text split file and return a list of sample names (one per line)."""
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names


def masks_to_boxes(mask_bin):
    """
    Convert a binary mask into instance bounding boxes via connected components.
    
    """
    H, W = mask_bin.shape
    num_labels, labels = cv2.connectedComponents(mask_bin.astype('uint8'))
    boxes = []
    for lab in range(1, num_labels):
        ys, xs = (labels == lab).nonzero()
        if ys.size == 0:
            continue
        # Half-open interval: add +1 to (x2, y2); normalization comes later.
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        boxes.append([x1, y1, x2, y2])
    return boxes, labels, num_labels


def xyxy_to_cxcywh_norm(boxes_xyxy, H, W):
    """Convert [x1, y1, x2, y2] boxes to normalized [cx, cy, w, h] in [0, 1]."""
    if len(boxes_xyxy) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    res = []
    for x1, y1, x2, y2 in boxes_xyxy:
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        res.append([cx, cy, w, h])
    return torch.tensor(res, dtype=torch.float32)


# -----------------------
# Augmentations (OpenCV)
# -----------------------
def _random_hflip(img, mask):
    """Horizontal flip with p=0.5; apply the same transform to image and mask."""
    if random.random() < 0.5:
        img  = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    return img, mask


def _random_vflip(img, mask):
    """Vertical flip with p=0.5; apply the same transform to image and mask."""
    if random.random() < 0.5:
        img  = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    return img, mask


def _random_rot90(img, mask):
    """Rotate by k * 90° with k∈{0,1,2,3} sampled once; keep arrays contiguous."""
    if random.random() < 0.5:
        k = random.randint(0, 3)
        if k > 0:
            img  = np.ascontiguousarray(np.rot90(img,  k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
    return img, mask


def _random_scale_and_crop(img, mask, out_size=1024, scale_min=0.75, scale_max=1.25):
    """
    Isotropic random scaling followed by a random crop to `out_size`.
    If the scaled image is smaller than `out_size`, pad by reflection first.
    """
    H, W = img.shape[:2]
    s = random.uniform(scale_min, scale_max)
    new_h, new_w = int(H * s), int(W * s)
    # Resize
    img_s  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # If smaller than target, reflect-pad to >= out_size
    pad_top = pad_bottom = pad_left = pad_right = 0
    if new_h < out_size:
        need = out_size - new_h
        pad_top = need // 2
        pad_bottom = need - pad_top
    if new_w < out_size:
        need = out_size - new_w
        pad_left = need // 2
        pad_right = need - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        img_s = cv2.copyMakeBorder(img_s, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_REFLECT_101)
        mask_s = cv2.copyMakeBorder(mask_s, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_REFLECT_101)
        new_h, new_w = img_s.shape[:2]
    # Random crop to out_size
    y0 = 0 if new_h == out_size else random.randint(0, new_h - out_size)
    x0 = 0 if new_w == out_size else random.randint(0, new_w - out_size)
    img_c  = img_s[y0:y0+out_size, x0:x0+out_size, :]
    mask_c = mask_s[y0:y0+out_size, x0:x0+out_size]
    return img_c, mask_c


def _random_brightness_contrast(img, max_bright=0.2, max_contrast=0.2):
    """
    Random brightness/contrast jitter (80% chance).
    Input can be uint8 or float32 in [0, 255]; output is float32 in [0, 255].
    """
    if random.random() < 0.8:
        alpha = 1.0 + random.uniform(-max_contrast, max_contrast)  # contrast
        beta  = 255.0 * random.uniform(-max_bright, max_bright)    # brightness
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255).astype(np.float32)
    return img


def _random_gaussian_blur(img, p=0.2):
    """Apply Gaussian blur with probability p using a small odd-sized kernel."""
    if random.random() < p:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def apply_augmentations(img, mask, aug_cfg):
    if aug_cfg is None:
        return img, mask

    # Spatial augmentations centered around 1024 resolution (downstream expects 1024)
    # 1) Geometry (apply consistently to both image and mask)
    if aug_cfg.get('hflip', True): img, mask = _random_hflip(img, mask)
    if aug_cfg.get('vflip', True): img, mask = _random_vflip(img, mask)
    if aug_cfg.get('rot90', True): img, mask = _random_rot90(img, mask)

    # 2) Scale jitter + random crop
    smin = aug_cfg.get('scale_min', 0.75)
    smax = aug_cfg.get('scale_max', 1.25)
    img, mask = _random_scale_and_crop(img, mask, out_size=aug_cfg.get('out_size', 1024),
                                       scale_min=smin, scale_max=smax)

    # 3) Photometric (image only)
    bc = aug_cfg.get('brightness_contrast', True)
    if bc:
        img = _random_brightness_contrast(
            img,
            max_bright=aug_cfg.get('max_bright', 0.2),
            max_contrast=aug_cfg.get('max_contrast', 0.2)
        )
    if aug_cfg.get('gaussian_blur', True):
        img = _random_gaussian_blur(img, p=aug_cfg.get('blur_p', 0.2))

    return img, mask


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, mask_size=(1024, 1024), augment=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_size = mask_size  # Target size (typically 1024, 1024)
        self.augment = augment

        allow = set(file_list)
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith('.png') and (f.replace('.png', '') in allow)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path  = os.path.join(self.mask_dir,  image_file)

        # Read image (BGR->RGB, float32 in [0,255])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Read mask (binary)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Resize to the canonical 1024 baseline (keeps augmentation logic simple)
        image = cv2.resize(image, (self.mask_size[1], self.mask_size[0]), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.mask_size[1], self.mask_size[0]), interpolation=cv2.INTER_NEAREST)

        # Training-time augmentation (pass augment=None for val/test)
        if self.augment is not None:
            image, mask = apply_augmentations(image, mask, self.augment)

        # Binarize mask explicitly to {0,1}
        mask_bin = (mask > 0).astype('uint8')

        # Compute instance masks and boxes on the augmented mask
        H, W = mask_bin.shape
        boxes_xyxy, labels_map, num_labels = masks_to_boxes(mask_bin)

        inst_masks = []
        for lab in range(1, num_labels):
            inst = (labels_map == lab).astype(np.float32)  # [H, W]
            if inst.sum() == 0:
                continue
            inst_masks.append(torch.from_numpy(inst))
        if len(inst_masks) > 0:
            inst_masks = torch.stack(inst_masks, dim=0)    # [N, H, W]
        else:
            inst_masks = torch.zeros((0, H, W), dtype=torch.float32)

        boxes_cxcywh = xyxy_to_cxcywh_norm(boxes_xyxy, H, W)
        labels = torch.zeros((boxes_cxcywh.shape[0],), dtype=torch.long)

        # To tensor (CHW for image; float mask)
        img_chw = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
        mask_tensor = torch.as_tensor(mask_bin.astype(np.float32), dtype=torch.float32)

        target = {
            'boxes': boxes_cxcywh,             # [N, 4] in [0, 1]
            'labels': labels,                  # [N]
            'orig_size': torch.tensor([H, W], dtype=torch.long),
            'inst_masks': inst_masks           # [N, H, W]
        }

        return img_chw, target, mask_tensor

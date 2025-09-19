# eval.py
import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from types import MethodType

from segment_anything import sam_model_registry
from lora import LoRA_sam
from config import CONFIG

# Use the same helper functions as in training
from utils import (
    initialize_metrics, accumulate_metrics,
    predict_masks_from_boxes, box_cxcywh_to_xyxy, nms_xyxy
)
from model import Prompt_Predictor


# ---------- helpers ----------
def set_seed(seed=42):
    """Make results deterministic where possible."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split_files(file_path):
    """Read a split file (one stem per line)."""
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names


def collate_fn_eval(batch):
    """Batch collator for evaluation: stack tensors and keep names."""
    imgs, masks, names = zip(*batch)
    imgs  = torch.stack(imgs,  dim=0)
    masks = torch.stack(masks, dim=0)
    return imgs, masks, list(names)


# ---------- dataset (read-only) ----------
class EvalDataset(Dataset):
    """Lightweight read-only dataset for evaluation."""
    def __init__(self, image_dir, mask_dir, id_list, size=(1024, 1024)):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.ids       = id_list
        self.size      = size

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img_path  = os.path.join(self.image_dir, f"{stem}.png")
        mask_path = os.path.join(self.mask_dir,  f"{stem}.png")

        img  = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: 
            raise FileNotFoundError(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img  = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [3, 1024, 1024]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            raise FileNotFoundError(mask_path)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask)  # [1024, 1024]

        return img, mask, stem


# ---------- SAM encoder extension ----------
def forward_inter(self, x: torch.Tensor):
    """Expose intermediate features from the image encoder."""
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    inter_features = []
    for blk in self.blocks:
        x = blk(x); inter_features.append(x)
    x = self.neck(x.permute(0, 3, 1, 2))
    return x, inter_features


def draw_mask_overlay(rgb_img_uint8, pred_mask_bin, gt_mask_bin, alpha=0.5):
    """RGB image overlaid with predicted mask (red) and ground truth mask (green)."""
    overlay = rgb_img_uint8.copy()
    # Predicted mask → RED
    red = np.zeros_like(overlay); red[..., 0] = 255  # RGB
    overlay = np.where(pred_mask_bin[..., None] > 0, (1 - alpha) * overlay + alpha * red, overlay).astype(np.uint8)
    # Ground-truth mask → GREEN
    green = np.zeros_like(overlay); green[..., 1] = 255
    overlay = np.where(gt_mask_bin[..., None] > 0, (1 - alpha) * overlay + alpha * green, overlay).astype(np.uint8)
    return overlay


def draw_gt_and_boxes(rgb_img_uint8, gt_mask_bin, boxes_xyxy, box_color=(255, 0, 0), alpha=0.5):
    """RGB image overlaid with GT mask (green) and predicted boxes (red)."""
    overlay = rgb_img_uint8.copy()
    # Ground-truth mask → GREEN
    green = np.zeros_like(overlay); green[..., 1] = 255
    overlay = np.where(gt_mask_bin[..., None] > 0, (1 - alpha) * overlay + alpha * green, overlay).astype(np.uint8)
    # Draw boxes (red)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (box_color[2], box_color[1], box_color[0]), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    set_seed(42)

    # Paths (align with train.py's eval_split)
    split = CONFIG.get('eval_split', 'val')  # 'val' or 'test'
    image_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'images')
    mask_dir  = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'masks')
    list_txt  = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], f'{split}.txt')

    # Logging
    log_dir  = os.path.join(CONFIG['log_dir_base'],  CONFIG['dataset_name'])
    save_dir = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'])
    os.makedirs(log_dir, exist_ok=True); os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, CONFIG.get('log_file', 'eval_metrics.log')),
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = CONFIG['device']
    img_size = int(CONFIG.get('img_size', 1024))

    # Unified filtering params (consistent with training-time evaluation)
    box_score_thresh = float(CONFIG.get('box_score_thresh', 0.5))
    nms_iou_thresh   = float(CONFIG.get('nms_iou_thresh', 0.5))
    max_dets         = int(CONFIG.get('max_dets', 10))
    mask_thresh      = float(CONFIG.get('mask_thresh', 0.5))  # Binarization for metrics

    # Output directory
    pred_out_dir = os.path.join(save_dir, f'pred_masks_{split}')
    os.makedirs(pred_out_dir, exist_ok=True)

    # ===== SAM + LoRA =====
    sam_model = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint'])
    sam_model.image_encoder.forward_inter = MethodType(forward_inter, sam_model.image_encoder)
    sam_model.to(device).eval()

    lora_layers = CONFIG.get('lora_layers', None)
    lora_sam_model = LoRA_sam(sam_model, rank=CONFIG.get('rank', 64), lora_layer=lora_layers).to(device).eval()

    lora_ckpt = os.path.join(save_dir, f"{CONFIG['save_prefix']}_lora.safetensors")
    if os.path.exists(lora_ckpt):
        try:
            lora_sam_model.load_lora_parameters(lora_ckpt)
            print(f"[OK] Loaded LoRA: {lora_ckpt}")
        except Exception as e:
            print(f"[ERR] Loading LoRA failed: {e}")
    else:
        print(f"[WARN] LoRA checkpoint not found: {lora_ckpt}")

    # ===== prompt predictor =====
    predictor = Prompt_Predictor(
        num_classes=1,
        num_queries=CONFIG.get('num_queries', 50),
        hidden_dim=CONFIG.get('hidden_dim', 256),
        nheads=CONFIG.get('nheads', 8),
        dec_layers=CONFIG.get('dec_layers', 6),
        dropout=CONFIG.get('dropout', 0.1),
    ).to(device).eval()

    pred_ckpt = os.path.join(save_dir, f"{CONFIG['save_prefix']}.pth")
    if not os.path.exists(pred_ckpt):
        print(f"[ERROR] Predictor checkpoint not found: {pred_ckpt}")
        return
    predictor.load_state_dict(torch.load(pred_ckpt, map_location=device))
    print(f"[OK] Loaded predictor: {pred_ckpt}")

    # ===== data =====
    ids = read_split_files(list_txt)
    dataset = EvalDataset(image_dir, mask_dir, ids, size=(img_size, img_size))
    loader  = DataLoader(dataset, batch_size=CONFIG.get('batch_size', 1),
                         shuffle=False, num_workers=0,
                         pin_memory=(device.type == 'cuda'),
                         collate_fn=collate_fn_eval)

    gm = initialize_metrics()

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc=f"Evaluating [{split}]"):
            # images: [B, 3, H, W] (CPU float32)
            images = images.to(device, non_blocking=True)

            # SAM preprocess on GPU per-sample
            pre_list = []
            for b in range(images.shape[0]):
                t = lora_sam_model.sam.preprocess(images[b])  # [1, 3, H, W]
                pre_list.append(t.squeeze(0))
            images_pre = torch.stack(pre_list, dim=0)        # [B, 3, H, W]

            # Encode
            image_embedding, _ = lora_sam_model.sam.image_encoder.forward_inter(images_pre)

            # Prompt prediction (keep inter_features=None for parity with train.py)
            outputs = predictor(image_embedding, None)
            logits = outputs['pred_logits']                     # [B, Q, K+1]

            # Foreground score: objectness = 1 - P(no-object)
            probs  = 1.0 - logits.softmax(-1)[..., -1]          # [B, Q]

            boxes  = outputs['pred_boxes']                      # [B, Q, 4] cxcywh in [0,1]
            # Scale to pixels and clamp to image bounds (consistent with train.py)
            boxes_xyxy_all = (box_cxcywh_to_xyxy(boxes) * float(img_size)).clamp(min=0.0, max=float(img_size))

            # ========= box selection: score threshold + NMS =========
            selected_boxes_batch = []
            B, Q, _ = boxes.shape
            for b in range(B):
                scores_b = probs[b]                 # [Q]
                boxes_b  = boxes_xyxy_all[b]       # [Q, 4]

                # Score threshold
                keep = scores_b > box_score_thresh
                boxes_f  = boxes_b[keep]
                scores_f = scores_b[keep]
                if boxes_f.numel() == 0:
                    selected_boxes_batch.append([])
                    continue

                # NMS
                keep_idx = nms_xyxy(boxes_f, scores_f, iou_thresh=nms_iou_thresh)
                keep_idx = keep_idx[:max_dets]
                sel = boxes_f[keep_idx].detach().cpu().tolist()
                selected_boxes_batch.append(sel)

            # SAM mask decoding (predict_masks_from_boxes also clamps internally)
            pred_masks = predict_masks_from_boxes(
                lora_sam_model.sam, image_embedding, selected_boxes_batch,
                device=device, out_size=img_size
            )  # [B, 1, H, W] logits
            preds_final = torch.sigmoid(pred_masks).cpu().numpy()

            gts = masks.numpy().astype(np.float32)  # [B, H, W]

            for bi in range(preds_final.shape[0]):
                name = names[bi]
                # Accumulate metrics with mask binarization threshold
                accumulate_metrics(preds_final[bi, 0], gts[bi], gm, threshold=mask_thresh)

                # Save binary mask
                bin_mask = (preds_final[bi, 0] > mask_thresh).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(pred_out_dir, f"{name}.png"), bin_mask)

                # Visualizations
                vis_rgb = images[bi].detach().cpu().permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
                gt_bin  = (gts[bi] > 0.5).astype(np.uint8)

                overlay_mask = draw_mask_overlay(vis_rgb, (bin_mask > 0).astype(np.uint8), gt_bin, alpha=0.5)
                cv2.imwrite(os.path.join(pred_out_dir, f"{name}_overlay_mask.png"),
                            cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

                overlay_boxes = draw_gt_and_boxes(vis_rgb, gt_bin, selected_boxes_batch[bi],
                                                  box_color=(255, 0, 0), alpha=0.5)
                cv2.imwrite(os.path.join(pred_out_dir, f"{name}_overlay_boxes.png"),
                            cv2.cvtColor(overlay_boxes, cv2.COLOR_RGB2BGR))

                # Save boxes as JSON (pixel coordinates, xyxy)
                meta = {"boxes_xyxy": selected_boxes_batch[bi]}
                with open(os.path.join(pred_out_dir, f"{name}.json"), "w") as f:
                    json.dump(meta, f)

    # Aggregate metrics
    tp, fp, fn = gm['tp'], gm['fp'], gm['fn']
    intersection, union = gm['intersection'], gm['union']
    iou = intersection / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    msg = (f"SAM Final Prediction Metrics [{split}] - "
           f"IoU: {iou:.4f}, Precision: {precision:.4f}, "
           f"Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info(msg); print(msg)

    print("\n=== Evaluation Finished ===")
    print(f"  Split: {split}")
    print(f"  IoU: {iou:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Saved to: {pred_out_dir}")


if __name__ == "__main__":
    main()

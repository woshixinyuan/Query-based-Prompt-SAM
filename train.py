# train.py
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm.auto import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from types import MethodType

from config import CONFIG, image_dir, mask_dir, train_txt, val_txt, test_txt
from datasets import SegmentationDataset, read_split_files
from model import Prompt_Predictor
from utils import (
    HungarianMatcher, SetCriterion, initialize_metrics, accumulate_metrics,
    predict_masks_from_boxes, box_cxcywh_to_xyxy, bce_dice_loss,
    box_xyxy_to_cxcywh, generalized_box_iou, nms_xyxy, mask_to_box
)
from segment_anything import sam_model_registry
from lora import LoRA_sam


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    """Make results deterministic where possible."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Batch collator for train/eval: stack image/mask, keep targets as list."""
    imgs, targets, masks = zip(*batch)
    imgs  = torch.stack(imgs,  dim=0)
    masks = torch.stack(masks, dim=0)
    return imgs, list(targets), masks


def setup_logging(log_dir: str, log_file: str):
    """Configure file logger once to avoid duplicate handlers on re-runs."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logger.setLevel(logging.INFO)


# -------------------------
# SAM: add forward_inter
# -------------------------
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


def main():
    set_seed(42)

    # -------------------------
    # Logging & dirs
    # -------------------------
    log_dir  = os.path.join(CONFIG['log_dir_base'],  CONFIG['dataset_name'])
    save_dir = os.path.join(CONFIG['save_dir_base'], CONFIG['dataset_name'])
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(log_dir, CONFIG['log_file'])

    # -------------------------
    # Build SAM + LoRA
    # -------------------------
    sam_model = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint'])
    sam_model.image_encoder.forward_inter = MethodType(forward_inter, sam_model.image_encoder)
    sam_model.to(CONFIG['device'])

    lora_sam_model = LoRA_sam(sam_model, rank=CONFIG['rank']).to(CONFIG['device'])

    # Freeze SAM; only train LoRA adapters + predictor
    for p in lora_sam_model.sam.parameters():
        p.requires_grad = False
    for layer in lora_sam_model.A_weights + lora_sam_model.B_weights:
        for p in layer.parameters():
            p.requires_grad = True

    # -------------------------
    # Data
    # -------------------------
    eval_split = CONFIG.get('eval_split', 'val')
    assert eval_split in ('val', 'test')

    train_files = read_split_files(train_txt)
    eval_txt_fp = val_txt if eval_split == 'val' else test_txt
    eval_files  = read_split_files(eval_txt_fp)

    default_augment = {
        'hflip': True, 'vflip': True, 'rot90': True,
        'scale_min': 0.75, 'scale_max': 1.25, 'out_size': CONFIG.get('img_size', 1024),
        'brightness_contrast': True, 'max_bright': 0.2, 'max_contrast': 0.2,
        'gaussian_blur': True, 'blur_p': 0.2
    }
    augment_cfg = CONFIG.get('augment', default_augment)

    train_dataset = SegmentationDataset(
        image_dir, mask_dir, train_files,
        mask_size=(CONFIG.get('img_size', 1024), CONFIG.get('img_size', 1024)),
        augment=augment_cfg
    )
    eval_dataset = SegmentationDataset(
        image_dir, mask_dir, eval_files,
        mask_size=(CONFIG.get('img_size', 1024), CONFIG.get('img_size', 1024)),
        augment=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(CONFIG['device'].type == 'cuda'),
        collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(CONFIG['device'].type == 'cuda'),
        collate_fn=collate_fn
    )

    # -------------------------
    # Prompt Predictor
    # -------------------------
    predictor = Prompt_Predictor(
        num_classes=1,
        num_queries=CONFIG.get('num_queries', 50),
        hidden_dim=CONFIG.get('hidden_dim', 256),
        nheads=CONFIG.get('nheads', 8),
        dec_layers=CONFIG.get('dec_layers', 6),
        dropout=CONFIG.get('dropout', 0.1),
    ).to(CONFIG['device'])
    for p in predictor.parameters():
        p.requires_grad = True

    # -------------------------
    # Optimizer & Criterion
    # -------------------------
    matcher = HungarianMatcher(
        cost_class=CONFIG.get('cost_class', 1.0),
        cost_bbox=CONFIG.get('cost_bbox', 5.0),
        cost_giou=CONFIG.get('cost_giou', 2.0),
        cost_mask=CONFIG.get('cost_mask', 0.0),
        img_size=CONFIG.get('img_size', 1024)
    )
    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        eos_coef=CONFIG.get('eos_coef', 0.1),
        loss_bbox=CONFIG.get('loss_bbox', 5.0),
        loss_giou=CONFIG.get('loss_giou', 2.0)
    ).to(CONFIG['device'])

    trainable_params = list(filter(lambda p: p.requires_grad,
                                   list(lora_sam_model.parameters()) + list(predictor.parameters())))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=CONFIG['learning_rate'],
        betas=CONFIG['betas'],
        weight_decay=CONFIG['weight_decay']
    )

    # -------------------------
    # Scheduler
    # -------------------------
    num_epochs = CONFIG['num_epochs']
    warmup_epochs = 3
    min_lr_factor = 0.01

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float((epoch + 1) / warmup_epochs)
        else:
            return float(min_lr_factor + (1 - min_lr_factor) * 0.5 *
                         (1 + math.cos((epoch - warmup_epochs) * math.pi / (num_epochs - warmup_epochs))))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------
    # Train/Eval
    # -------------------------
    best_iou = float('-inf')
    best_epoch = 0

    img_size = CONFIG.get('img_size', 1024)
    box_score_thresh = CONFIG.get('box_score_thresh', 0.5)
    nms_iou_thresh   = CONFIG.get('nms_iou_thresh', 0.5)
    max_dets         = CONFIG.get('max_dets', 10)

    lambda_mask        = CONFIG.get('lambda_mask', 1.0)
    bce_w              = CONFIG.get('bce_weight', 1.0)
    dice_w             = CONFIG.get('dice_weight', 1.0)
    lambda_consistency = CONFIG.get('lambda_consistency', 1.0)
    lambda_fullmask    = CONFIG.get('lambda_fullmask', 1.0)

    mask_thresh = CONFIG.get('mask_thresh', 0.5)

    for epoch in range(num_epochs):
        lora_sam_model.train()
        predictor.train()

        total_loss = total_batches = 0
        train_loss_ce = train_loss_bbox = train_loss_giou = 0.0
        train_loss_mask = train_loss_cons = train_loss_fullmask = 0.0

        # TRAIN
        for images, targets, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            dynamic_ncols=True,
            leave=True                     
        ):
            images = images.to(CONFIG['device'], non_blocking=True)
            masks  = masks.to(CONFIG['device'],  non_blocking=True).float()

            # SAM preprocess on GPU
            pre_list = []
            with torch.no_grad():
                for b in range(images.shape[0]):
                    t = lora_sam_model.sam.preprocess(images[b])
                    pre_list.append(t.squeeze(0))
            images_pre = torch.stack(pre_list, dim=0)

            # targets → device
            dev_targets = []
            for t in targets:
                dt = {
                    'boxes': t['boxes'].to(CONFIG['device']),
                    'labels': t['labels'].to(CONFIG['device']),
                    'orig_size': t['orig_size'].to(CONFIG['device']),
                }
                if t.get('inst_masks', None) is not None:
                    dt['inst_masks'] = t['inst_masks'].to(CONFIG['device']).float()
                dev_targets.append(dt)

            # Forward
            image_embedding, _ = lora_sam_model.sam.image_encoder.forward_inter(images_pre)
            outputs = predictor(image_embedding, None)

            # (1) Hungarian match + CE/L1/GIoU
            loss_dict = criterion(
                outputs, dev_targets,
                sam_model=lora_sam_model.sam,
                image_embeddings=image_embedding
            )
            loss_set = loss_dict['loss']
            indices = loss_dict['indices']

            # (2) Mask supervision & consistency
            loss_mask_terms, loss_cons_terms = [], []
            for b, (src_i, tgt_i) in enumerate(indices):
                if len(src_i) == 0:
                    continue
                boxes_cxcywh = outputs['pred_boxes'][b, src_i]
                boxes_xyxy   = (box_cxcywh_to_xyxy(boxes_cxcywh) * img_size).clamp_(0.0, float(img_size))

                gt_inst = dev_targets[b].get('inst_masks', None)
                if gt_inst is None or gt_inst.numel() == 0:
                    continue
                gt_pair_masks = gt_inst[tgt_i]

                curr_embed = image_embedding[b:b+1]
                sparse, dense_prompt = lora_sam_model.sam.prompt_encoder(points=None, boxes=boxes_xyxy, masks=None)
                low_res_masks, _ = lora_sam_model.sam.mask_decoder(
                    image_embeddings=curr_embed,
                    image_pe=lora_sam_model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense_prompt,
                    multimask_output=False,
                )
                up = torch.nn.functional.interpolate(
                    low_res_masks, size=(img_size, img_size), mode='bilinear', align_corners=False
                )  # [M,1,H,W]

                # Mask supervision
                gt_pair_masks_ = gt_pair_masks.unsqueeze(1)
                loss_m = bce_dice_loss(up, gt_pair_masks_, bce_w=bce_w, dice_w=dice_w)
                loss_mask_terms.append(loss_m)

                # Consistency
                probs = up.sigmoid().squeeze(1).detach()
                pred_boxes_from_mask = torch.stack([mask_to_box(p, thr=0.5) for p in probs], dim=0).to(up.device)
                cxcywh_from_mask = box_xyxy_to_cxcywh(pred_boxes_from_mask / img_size)

                loss_cons_l1 = torch.nn.functional.l1_loss(cxcywh_from_mask, boxes_cxcywh, reduction='mean')
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(cxcywh_from_mask),
                    box_cxcywh_to_xyxy(boxes_cxcywh)
                )
                loss_cons_giou = (1. - giou).mean()
                loss_cons_terms.append(loss_cons_l1 + loss_cons_giou)

            loss_mask = torch.stack(loss_mask_terms).mean() if len(loss_mask_terms) > 0 else torch.tensor(0., device=loss_set.device)
            loss_cons = torch.stack(loss_cons_terms).mean() if len(loss_cons_terms) > 0 else torch.tensor(0., device=loss_set.device)

            # (2b) Full-image mask supervision (score threshold + NMS + merge)
            fullmask_terms = []
            probs_all  = outputs['pred_logits'].softmax(-1)
            logits_all = 1.0 - probs_all[..., -1]  # objectness
            boxes_all  = box_cxcywh_to_xyxy(outputs['pred_boxes']) * img_size

            B, Q, _ = boxes_all.shape
            for b in range(B):
                scores_b = logits_all[b]
                boxes_b  = boxes_all[b]
                keep = scores_b > box_score_thresh
                boxes_f  = boxes_b[keep]
                scores_f = scores_b[keep]
                if boxes_f.numel() == 0:
                    merged_logits = torch.zeros((1, img_size, img_size), device=images.device)
                else:
                    keep_idx = nms_xyxy(boxes_f, scores_f, iou_thresh=nms_iou_thresh)
                    keep_idx = keep_idx[:max_dets]
                    sel = boxes_f[keep_idx].clamp(0.0, float(img_size))

                    curr_embed = image_embedding[b:b+1]
                    sparse, dense_prompt = lora_sam_model.sam.prompt_encoder(points=None, boxes=sel, masks=None)
                    low_res_masks, _ = lora_sam_model.sam.mask_decoder(
                        image_embeddings=curr_embed,
                        image_pe=lora_sam_model.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense_prompt,
                        multimask_output=False,
                    )
                    up = torch.nn.functional.interpolate(
                        low_res_masks, size=(img_size, img_size), mode='bilinear', align_corners=False
                    )
                    merged_logits = up.max(dim=0)[0]  # [1,H,W]

                gt_full = masks[b].unsqueeze(0)
                loss_full = bce_dice_loss(merged_logits, gt_full, bce_w=bce_w, dice_w=dice_w)
                fullmask_terms.append(loss_full)

            loss_fullmask = torch.stack(fullmask_terms).mean() if len(fullmask_terms) > 0 else torch.tensor(0., device=loss_set.device)

            # Total loss
            loss = loss_set + lambda_mask * loss_mask + lambda_consistency * loss_cons + lambda_fullmask * loss_fullmask

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            train_loss_ce    += loss_dict['loss_ce'].item()
            train_loss_bbox  += loss_dict['loss_bbox'].item()
            train_loss_giou  += loss_dict['loss_giou'].item()
            train_loss_mask  += loss_mask.item()
            train_loss_cons  += loss_cons.item()
            train_loss_fullmask += loss_fullmask.item()
            total_batches += 1

        # Epoch train averages
        avg_train_loss = total_loss / max(1, total_batches)
        avg_train_ce   = train_loss_ce / max(1, total_batches)
        avg_train_bbox = train_loss_bbox / max(1, total_batches)
        avg_train_giou = train_loss_giou / max(1, total_batches)
        avg_train_mask = train_loss_mask / max(1, total_batches)
        avg_train_cons = train_loss_cons / max(1, total_batches)
        avg_train_full = train_loss_fullmask / max(1, total_batches)

        # EVAL
        lora_sam_model.eval(); predictor.eval()
        eval_loss_total = 0.0
        num_eval_batches = 0
        gm = initialize_metrics()

        with torch.no_grad():
            for images, targets, masks in tqdm(
                eval_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Eval:{eval_split}]",
                dynamic_ncols=True,
                leave=False
            ):
                images = images.to(CONFIG['device'], non_blocking=True)
                masks  = masks.to(CONFIG['device'],  non_blocking=True).float()

                pre_list = []
                for b in range(images.shape[0]):
                    t = lora_sam_model.sam.preprocess(images[b])
                    pre_list.append(t.squeeze(0))
                images_pre = torch.stack(pre_list, dim=0)

                dev_targets = []
                for t in targets:
                    dt = {
                        'boxes': t['boxes'].to(CONFIG['device']),
                        'labels': t['labels'].to(CONFIG['device']),
                        'orig_size': t['orig_size'].to(CONFIG['device']),
                    }
                    if t.get('inst_masks', None) is not None:
                        dt['inst_masks'] = t['inst_masks'].to(CONFIG['device']).float()
                    dev_targets.append(dt)

                image_embedding, _ = lora_sam_model.sam.image_encoder.forward_inter(images_pre)
                outputs = predictor(image_embedding, None)

                loss_dict = criterion(
                    outputs, dev_targets,
                    sam_model=lora_sam_model.sam,
                    image_embeddings=image_embedding
                )
                eval_loss_total += loss_dict['loss'].item()
                num_eval_batches += 1

                logits = outputs['pred_logits']
                probs  = 1.0 - logits.softmax(-1)[..., -1]
                boxes  = outputs['pred_boxes']
                boxes_xyxy_all = (box_cxcywh_to_xyxy(boxes) * img_size).clamp(0.0, float(img_size))

                selected_boxes_batch = []
                B, Q, _ = boxes.shape
                for b in range(B):
                    scores_b = probs[b]
                    boxes_b  = boxes_xyxy_all[b]
                    keep = scores_b > box_score_thresh
                    boxes_f  = boxes_b[keep]
                    scores_f = scores_b[keep]
                    if boxes_f.numel() == 0:
                        selected_boxes_batch.append([])
                        continue
                    keep_idx = nms_xyxy(boxes_f, scores_f, iou_thresh=nms_iou_thresh)
                    keep_idx = keep_idx[:max_dets]
                    sel = boxes_f[keep_idx].detach().cpu().tolist()
                    selected_boxes_batch.append(sel)

                pred_masks = predict_masks_from_boxes(
                    lora_sam_model.sam, image_embedding, selected_boxes_batch,
                    device=CONFIG['device'], out_size=img_size
                )
                preds_final = torch.sigmoid(pred_masks).cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                if masks_np.ndim == 3:
                    masks_np = masks_np[:, None, ...]
                for p_final, m_gt in zip(preds_final, masks_np):
                    accumulate_metrics(p_final[0], m_gt[0], gm, threshold=mask_thresh)

        # Aggregate metrics
        tp, fp, fn = gm['tp'], gm['fp'], gm['fn']
        intersection, union = gm['intersection'], gm['union']
        iou = intersection / (union + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        avg_eval_loss = eval_loss_total / max(1, num_eval_batches)

        composite_score = (iou * 0.25 + f1 * 0.25 + precision * 0.25 + recall * 0.25)
        lr_now = optimizer.param_groups[0]['lr']

        log_message = (
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"LR: {lr_now:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} "
            f"(CE {avg_train_ce:.4f}, BBox {avg_train_bbox:.4f}, GIoU {avg_train_giou:.4f}, "
            f"Mask {avg_train_mask:.4f}, Cons {avg_train_cons:.4f}, FullMask {avg_train_full:.4f}) | "
            f"Eval({eval_split}) Loss: {avg_eval_loss:.4f} | "
            f"IoU: {iou:.4f}, F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}, "
            f"Composite: {composite_score:.4f}"
        )
        print(log_message)           
        logging.info(log_message)

        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch + 1
            lora_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}_lora.safetensors")
            ckpt_path = os.path.join(save_dir, f"{CONFIG['save_prefix']}.pth")
            if hasattr(lora_sam_model, 'save_lora_parameters'):
                lora_sam_model.save_lora_parameters(lora_path)
            torch.save(predictor.state_dict(), ckpt_path)
            best_msg = (f"[BEST by IoU] Saved at epoch {best_epoch} on [{eval_split}] "
                        f"with IoU {best_iou:.4f} (Composite {composite_score:.4f})")
            print(best_msg)
            logging.info(best_msg)

        scheduler.step()

    logging.info("Training completed")
    print("Training completed")


if __name__ == "__main__":
    main()

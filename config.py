import os
import torch

CONFIG = {
    'dataset_name': 'xian',
    'data_base_dir': 'datasets',
    'log_dir_base': 'logs',
    'log_file': 'best_model_metrics.log',
    'save_dir_base': 'logs',
    'save_prefix': 'best_model',

    # SAM backbone
    'model_type': 'vit_l',
    'checkpoint': 'weights/sam_vit_l_0b3195.pth',
    'img_size': 1024,

    # Device & training
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-4,

    # Evaluation split: 'val' or 'test'
    'eval_split': 'test',

    # LoRA hyper-parameters
    'rank': 256,

    # Batch size
    'batch_size': 1,

    # ========= Prompt_Predictor =========
    'num_queries': 5,       
    'hidden_dim': 256,
    'nheads': 8,
    'dec_layers': 6,
    'dropout': 0.1,

    # Matching cost weights (Hungarian)
    'eos_coef': 0.1,             # Penalty for background to suppress false positives
    'cost_class': 1.0,
    'cost_bbox': 5.0,
    'cost_giou': 2.0,
    'cost_mask': 2.0,            # Soft IoU cost for mask matching

    # Loss weights during training
    'loss_bbox': 5.0,
    'loss_giou': 2.0,
    'lambda_mask': 1.0,          # Instance-level mask supervision
    'bce_weight': 1.0,           # BCE loss weight for masks
    'dice_weight': 1.0,          # Dice loss weight for masks
    'lambda_consistency': 1.0,   # Consistency loss between mask and box
    'lambda_fullmask': 1.0,      # Full-image mask supervision (merged after NMS)

    # Inference / evaluation filtering parameters
    'nms_iou_thresh': 0.5,
    'max_dets': 5,
    'box_score_thresh': 0.5,

    # Threshold for binarizing masks (used for metrics)
    'mask_thresh': 0.5,

    # Save predictions (used in eval.py)
    'save_predictions': True,
    'pred_save_dir': 'logs/xian/pred_masks',
    'pred_vis_dir':  'logs/xian/pred_vis',

    # ========= Data augmentation (applied to training set only) =========
    'augment': {
        'hflip': True,
        'vflip': True,
        'rot90': True,
        'scale_min': 0.75,
        'scale_max': 1.25,
        'out_size': 1024,              # Usually kept consistent with img_size
        'brightness_contrast': True,
        'max_bright': 0.2,
        'max_contrast': 0.2,
        'gaussian_blur': True,
        'blur_p': 0.2,
    },
}

# Paths
image_dir = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'images')
mask_dir  = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'masks')
train_txt = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'train.txt')
val_txt   = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'val.txt')
test_txt  = os.path.join(CONFIG['data_base_dir'], CONFIG['dataset_name'], 'test.txt')

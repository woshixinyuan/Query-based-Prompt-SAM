# QP-SAM: Query-based Prompt Generation for Segment Anything in Urban-Village Identification

> Parameter-efficient domain adaptation of SAM by **learning prompts**, with frozen SAM weights and lightweight LoRA adapters.

---

## 🔹 Highlights

* **Query-based prompt predictor** generates a small set of **box prompts** from SAM image features; SAM’s prompt encoder & mask decoder remain intact.
* **Frozen SAM** + **LoRA** on the image encoder for efficient adaptation; transformer decoder uses **learnable queries**.
* **Hungarian matching** for instance-level assignment, plus **mask–box consistency** and **global full-image supervision** for coherent predictions.
* Strong cross-domain results on **Beijing** and **Xi’an** urban-village datasets (see results below).&#x20;

---

## 📦 Environment

```text
Python >= 3.10
PyTorch >= 2.1 (with CUDA if using GPU)
torchvision, numpy, opencv-python, tqdm, safetensors, pyyaml, matplotlib
```

Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Adjust torch/torchvision versions to your local CUDA.

---

## 📂 Data layout

```
datasets/
  <dataset_name>/
    images/          # RGB .png
    masks/           # binary masks (.png), same file names as images
    train.txt        # one file name per line (with extension)
    val.txt
    test.txt
```

---

## 🔧 Weights

Download **SAM ViT-L** checkpoint:

```bash
mkdir -p weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
     -O weights/sam_vit_l_0b3195.pth
```

---

## 🚀 Quick Start

```bash
# 1) create env & install (see above)

# 2) train (single GPU)
python train.py

# 3) evaluate on the split defined by CONFIG['eval_split']
python eval.py
```

Artifacts:

* `logs/<dataset_name>/best_model.pth` – prompt predictor weights
* `logs/<dataset_name>/best_model_lora.safetensors` – LoRA adapters
* `logs/<dataset_name>/<log_file>` – metrics log
* `logs/<dataset_name>/pred_masks_<split>/` – evaluation outputs (binary masks, overlays, boxes & JSON)

---

## ⚙️ Configuration (excerpt from `config.py`)

```python
CONFIG = {
    'dataset_name': 'xian',
    'data_base_dir': 'datasets',
    'log_dir_base': 'logs',
    'save_dir_base': 'logs',
    'save_prefix': 'best_model',
    'log_file': 'best_model_metrics.log',

    # SAM backbone
    'model_type': 'vit_l',
    'checkpoint': 'weights/sam_vit_l_0b3195.pth',
    'img_size': 1024,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # LoRA
    'rank': 256,  # set to 1 for minimal-parameter variant

    # Prompt predictor (transformer decoder)
    'num_queries': 5,
    'hidden_dim': 256,
    'nheads': 8,
    'dec_layers': 6,
    'dropout': 0.1,

    # Hungarian matching costs
    'eos_coef': 0.1,
    'cost_class': 1.0,
    'cost_bbox': 5.0,
    'cost_giou': 2.0,
    'cost_mask': 2.0,

    # Loss weights (train.py)
    'loss_bbox': 5.0,
    'loss_giou': 2.0,
    'lambda_mask': 1.0,
    'bce_weight': 1.0,
    'dice_weight': 1.0,
    'lambda_consistency': 1.0,
    'lambda_fullmask': 1.0,

    # Inference
    'nms_iou_thresh': 0.5,
    'max_dets': 5,
    'box_score_thresh': 0.5,
    'mask_thresh': 0.5,

    # Data & training
    'batch_size': 1,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-4,
    'eval_split': 'test',

    # Optional data augmentation (train only)
    'augment': {
        'hflip': True, 'vflip': True, 'rot90': True,
        'scale_min': 0.75, 'scale_max': 1.25, 'out_size': 1024,
        'brightness_contrast': True, 'max_bright': 0.2, 'max_contrast': 0.2,
        'gaussian_blur': True, 'blur_p': 0.2,
    },
}
```

**Tips**

* Set `rank=1` to try the minimal-parameter LoRA variant; increase `num_queries` when images contain many instances.
* `eval_split` controls whether `val.txt` or `test.txt` is used in `eval.py`.

---

## 🧠 Method

The model learns **a small set of box prompts** from SAM’s frozen image features. A transformer decoder with **learnable queries** interacts with positional-encoded features to produce class logits (foreground/background) and normalized boxes. During training, **Hungarian matching** assigns predictions to ground-truth instances using a composite cost (classification, L1, GIoU, mask consistency). To avoid fragmented outputs, we also supervise a **global full-image mask** obtained by decoding the selected prompts with SAM and fusing them via pixel-wise max. At inference, we select prompts by score thresholding + **NMS**, decode with SAM, and fuse the masks.&#x20;

---

## 📊 Results

**Overall performance** on Beijing / Xi’an (IoU / F1):

* **QP-SAM (ours):** **75.1 / 81.4 IoU**, **85.8 / 89.7 F1**
---

## 🧪 Reproduce

1. Prepare data as shown above and download the SAM ViT-L checkpoint.
2. Adjust `CONFIG` for your dataset (notably `dataset_name`, `eval_split`, and LoRA `rank`).
3. Run:

```bash
python train.py
python eval.py
```

4. Inspect `logs/<dataset_name>/pred_masks_<split>/` for binary masks and overlays; metrics are printed and logged.

---

## ✍️ Citation

If you find this repository helpful, please cite:


---

## 🙏 Acknowledgements

* [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) for the backbone and pretrained weights.
* [UV-SAM](https://github.com/tsinghua-fib-lab/UV-SAM) for a strong adaptation baseline and dataset preparation guidance.

---

## 📜 License

Code in this repository is released under the **MIT License** (see `LICENSE`). SAM weights are provided by their respective authors under their licenses.

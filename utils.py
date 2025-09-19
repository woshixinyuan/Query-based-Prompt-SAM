import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============ box ops ============

def box_cxcywh_to_xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = x.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1).clamp(min=0)
    h  = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)

def box_area(boxes):
    """Compute area for [x1, y1, x2, y2] boxes."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)

def box_iou(boxes1, boxes2):
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """Pairwise Generalized IoU for [x1, y1, x2, y2] boxes."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou

# ============ mask losses & utils ============

def bce_dice_loss(logits, target, bce_w=1.0, dice_w=1.0, eps=1e-6):
    """
    Combined BCE-with-logits + Dice loss.

    Parameters
    ----------
    logits : Tensor
        [B,1,H,W] or [1,H,W] raw logits.
    target : Tensor
        Same shape as logits, float {0,1}.
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1. - (2 * inter + eps) / (union + eps)
    dice = dice.mean()

    return bce_w * bce + dice_w * dice

def soft_iou_matrix(pred_probs, gt_masks, eps=1e-6):
    """
    Soft IoU between predicted probability maps and binary GT masks.

    Parameters
    ----------
    pred_probs : Tensor
        [Q,H,W] in [0,1].
    gt_masks : Tensor
        [N,H,W] in {0,1}.

    Returns
    -------
    Tensor
        [Q,N] matrix of soft IoU.
    """
    Q, H, W = pred_probs.shape
    N, h, w = gt_masks.shape
    assert (H, W) == (h, w)

    P = pred_probs.view(Q, -1)                 # [Q,HW]
    G = gt_masks.view(N, -1)                   # [N,HW]
    inter = (P[:, None, :] * G[None, :, :]).sum(-1)  # [Q,N]
    union = P.sum(-1, keepdim=True) + G.sum(-1)[None, :] - inter
    iou = inter / (union + eps)
    return iou  # [Q,N]

def mask_to_box(prob, thr=0.5):
    """
    Hard-threshold box from a probability map (non-differentiable).

    Parameters
    ----------
    prob : Tensor
        [H,W] or [1,H,W], float in [0,1].
    thr : float
        Threshold for binarization.

    Returns
    -------
    Tensor
        [4] xyxy pixel coordinates in a half-open interval [x1,y1,x2,y2).
    """
    if prob.dim() == 3:
        prob = prob[0]
    H, W = prob.shape[-2], prob.shape[-1]
    m = (prob >= thr).float()
    ys, xs = torch.where(m > 0.5)
    if ys.numel() == 0:
        return torch.tensor([0., 0., 0., 0.], device=prob.device, dtype=prob.dtype)
    x1, x2 = xs.min().float(), xs.max().float() + 1.0
    y1, y2 = ys.min().float(), ys.max().float() + 1.0
    # Clip to image bounds (half-open interval)
    x1 = x1.clamp(0.0, float(W))
    y1 = y1.clamp(0.0, float(H))
    x2 = x2.clamp(0.0, float(W))
    y2 = y2.clamp(0.0, float(H))
    return torch.stack([x1, y1, x2, y2], dim=0)

def soft_mask_to_box(prob_map, eps=1e-6):
    """
    Differentiable "soft box" from probability maps via moments.

    Parameters
    ----------
    prob_map : Tensor
        [M,H,W] in [0,1].

    Returns
    -------
    Tensor
        [M,4] xyxy pixel coordinates (clipped, half-open interval).
    """
    assert prob_map.dim() == 3, "prob_map should be [M,H,W]"
    M, H, W = prob_map.shape
    device = prob_map.device
    dtype  = prob_map.dtype

    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    yy = ys.view(1, H, 1).expand(M, H, W)
    xx = xs.view(1, 1, W).expand(M, H, W)

    mass = prob_map.sum(dim=(1, 2)) + eps              # [M]
    cx = (prob_map * xx).sum(dim=(1, 2)) / mass        # [M]
    cy = (prob_map * yy).sum(dim=(1, 2)) / mass        # [M]

    var_x = (prob_map * (xx - cx.view(M, 1, 1))**2).sum(dim=(1, 2)) / mass
    var_y = (prob_map * (yy - cy.view(M, 1, 1))**2).sum(dim=(1, 2)) / mass

    # Width/height via variance (uniform box assumption)
    w = (12.0 * var_x).clamp(min=eps).sqrt()           # [M]
    h = (12.0 * var_y).clamp(min=eps).sqrt()           # [M]

    # Clip to image bounds (half-open interval)
    x1 = (cx - 0.5 * w).clamp(0.0, float(W))
    y1 = (cy - 0.5 * h).clamp(0.0, float(H))
    x2 = (cx + 0.5 * w).clamp(0.0, float(W))
    y2 = (cy + 0.5 * h).clamp(0.0, float(H))

    return torch.stack([x1, y1, x2, y2], dim=1)  # [M,4]

# ============ NMS for inference ============

def nms_xyxy(boxes, scores, iou_thresh=0.5):
    """
    Greedy NMS on xyxy boxes.

    Parameters
    ----------
    boxes : Tensor
        [N,4] xyxy.
    scores : Tensor
        [N].
    iou_thresh : float
        IoU threshold.

    Returns
    -------
    Tensor
        LongTensor indices to keep (on the same device).
    """
    if boxes.numel() == 0:
        return torch.as_tensor([], dtype=torch.long, device=boxes.device)
    keep = []
    idxs = scores.sort(descending=True).indices
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        iou, _ = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
        idxs = idxs[1:][iou.squeeze(0) <= iou_thresh]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

# ============ matcher ============

try:
    from scipy.optimize import linear_sum_assignment
    _has_scipy = True
except Exception:
    _has_scipy = False

class HungarianMatcher(nn.Module):
    """
    Hungarian matching with optional mask cost (soft IoU via SAM decoding).

    If `sam_model` and `image_embeddings` and per-instance GT masks are provided,
    an additional mask cost is computed and added to the cost matrix.
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 cost_mask=0.0, img_size=1024):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou
        self.cost_mask  = cost_mask
        self.img_size   = img_size

    @torch.no_grad()
    def forward(self, outputs, targets, sam_model=None, image_embeddings=None, gt_masks=None):
        """
        Parameters
        ----------
        outputs : dict
            'pred_logits' [B,Q,K+1], 'pred_boxes' [B,Q,4] (cxcywh in [0,1]).
        targets : list of dict
            Each with keys 'boxes', 'labels', and optional 'inst_masks'.
        sam_model : optional
            SAM model for mask decoding (used if cost_mask > 0).
        image_embeddings : optional
            Image embeddings [B,256,H',W'] aligned with `sam_model`.
        gt_masks : unused
            Kept for signature compatibility.

        Returns
        -------
        list[Tuple[Tensor, Tensor]]
            Per-batch indices (src, tgt) after Hungarian assignment.
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].softmax(-1)
        out_bbox = outputs['pred_boxes']

        indices = []
        device_idx = out_bbox.device
        for b in range(bs):
            tgt_ids  = targets[b]['labels']                  # [N]
            tgt_bbox = targets[b]['boxes']                   # [N,4]
            inst_masks = targets[b].get('inst_masks', None)  # [N,H,W] or None
            N = tgt_bbox.shape[0]
            if N == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64, device=device_idx),
                                torch.as_tensor([], dtype=torch.int64, device=device_idx)))
                continue

            # class / bbox costs
            cost_class = -out_prob[b][:, tgt_ids].detach().cpu()            # [Q,N]
            pb = out_bbox[b]
            tb = tgt_bbox
            pb_xyxy = box_cxcywh_to_xyxy(pb)
            tb_xyxy = box_cxcywh_to_xyxy(tb)

            cost_bbox = torch.cdist(pb, tb, p=1).detach().cpu()             # [Q,N]
            giou = generalized_box_iou(pb_xyxy, tb_xyxy).detach().cpu()     # [Q,N]
            cost_giou = (1.0 - giou)

            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            # optional mask cost
            if self.cost_mask > 0 and sam_model is not None and image_embeddings is not None and inst_masks is not None and inst_masks.numel() > 0:
                boxes_xyxy_pix = (pb_xyxy * self.img_size).to(image_embeddings.device)
                # clip to image bounds (half-open interval)
                boxes_xyxy_pix = boxes_xyxy_pix.clamp_(min=0.0, max=float(self.img_size))
                curr_embed = image_embeddings[b:b+1]  # [1,256,64,64]

                sparse, dense = sam_model.prompt_encoder(points=None, boxes=boxes_xyxy_pix, masks=None)
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=curr_embed,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )  # [Q,1,h,w]
                up = F.interpolate(low_res_masks, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)  # [Q,1,H,W]
                pred_probs = up.sigmoid().squeeze(1).detach().cpu()  # [Q,H,W]

                gt_stack = inst_masks.detach().cpu()                 # [N,H,W]
                iou = soft_iou_matrix(pred_probs, gt_stack)          # [Q,N]
                cost_mask = (1.0 - iou)
                C = C + self.cost_mask * cost_mask

            # Hungarian assignment
            C = C.numpy()
            if _has_scipy:
                row, col = linear_sum_assignment(C)
                i = torch.as_tensor(row, dtype=torch.int64, device=device_idx)
                j = torch.as_tensor(col, dtype=torch.int64, device=device_idx)
            else:
                # greedy fallback without SciPy
                Qn, Nn = C.shape
                used_r, used_c = set(), set()
                pairs = []
                flat = [(C[r, c], r, c) for r in range(Qn) for c in range(Nn)]
                flat.sort()
                for _, r, c in flat:
                    if r in used_r or c in used_c:
                        continue
                    used_r.add(r); used_c.add(c)
                    pairs.append((r, c))
                    if len(pairs) == min(Qn, Nn):
                        break
                if len(pairs) == 0:
                    i = torch.as_tensor([], dtype=torch.int64, device=device_idx)
                    j = torch.as_tensor([], dtype=torch.int64, device=device_idx)
                else:
                    rr, cc = zip(*pairs)
                    i = torch.as_tensor(rr, dtype=torch.int64, device=device_idx)
                    j = torch.as_tensor(cc, dtype=torch.int64, device=device_idx)

            indices.append((i, j))
        return indices

# ============ criterion ============

class SetCriterion(nn.Module):
    """Compute classification (CE) and box (L1 + GIoU) losses given assignments."""
    def __init__(self, num_classes, matcher, eos_coef=0.1, loss_bbox=5.0, loss_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_bbox_w = loss_bbox
        self.loss_giou_w = loss_giou

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']  # [B,Q,K+1]
        bs, Q, _ = src_logits.shape
        device = src_logits.device

        target_classes = torch.full((bs, Q), self.num_classes, dtype=torch.long, device=device)
        for b, (src_i, tgt_i) in enumerate(indices):
            if len(src_i) > 0:
                target_classes[b, src_i] = targets[b]['labels'][tgt_i]

        loss_ce = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.empty_weight)
        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        src_boxes = outputs['pred_boxes']       # [B,Q,4]
        device = src_boxes.device

        src_list, tgt_list = [], []
        for b, (src_i, tgt_i) in enumerate(indices):
            if len(src_i) == 0:
                continue
            src_list.append(src_boxes[b, src_i])
            tgt_list.append(targets[b]['boxes'][tgt_i])
        if len(src_list) == 0:
            z = torch.tensor(0., device=device)
            return z, z

        src_cat = torch.cat(src_list, dim=0)
        tgt_cat = torch.cat(tgt_list, dim=0)

        loss_bbox = F.l1_loss(src_cat, tgt_cat, reduction='mean')
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_cat), box_cxcywh_to_xyxy(tgt_cat))
        loss_giou = (1. - giou).mean()
        return loss_bbox, loss_giou

    def forward(self, outputs, targets, **kwargs):
        indices = self.matcher(outputs, targets, **kwargs)
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
        loss_bbox = loss_bbox * self.loss_bbox_w
        loss_giou = loss_giou * self.loss_giou_w

        total = loss_ce + loss_bbox + loss_giou
        return {
            'loss': total,
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'indices': indices
        }

# ============ metrics ============

def initialize_metrics():
    """Return a dict accumulator for segmentation metrics."""
    return {'tp': 0, 'fp': 0, 'fn': 0, 'intersection': 0, 'union': 0}

def accumulate_metrics(preds, targets, global_metrics, threshold=0.5):
    """Accumulate TP/FP/FN and IoU numerators/denominators using a thresholded mask."""
    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = (targets > threshold).astype(np.uint8)
    tp = np.logical_and(preds_binary == 1, targets_binary == 1).sum()
    fp = np.logical_and(preds_binary == 1, targets_binary == 0).sum()
    fn = np.logical_and(preds_binary == 0, targets_binary == 1).sum()
    intersection = tp
    union = np.logical_or(preds_binary, targets_binary).sum()
    gm = global_metrics
    gm['tp'] += tp; gm['fp'] += fp; gm['fn'] += fn; gm['intersection'] += intersection; gm['union'] += union

# ============ SAM decoding (boxes → masks for inference) ============

def predict_masks_from_boxes(sam_model, image_embeddings, boxes_batch, device='cuda', out_size=1024):
    """
    Decode masks from per-image lists of xyxy boxes.

    Parameters
    ----------
    sam_model : nn.Module
        SAM model with prompt encoder and mask decoder.
    image_embeddings : Tensor
        [B, C, H', W'] encoder features.
    boxes_batch : List[List[List[float]]]
        Per-image list of xyxy boxes in pixel coordinates.
    device : str
        Target device for decoding.
    out_size : int
        Output spatial size (H=W).

    Returns
    -------
    Tensor
        [B, 1, H, W] logits; multiple instances are merged by max.
    """
    sam_model.eval()
    final_predictions = []
    B = image_embeddings.shape[0]
    with torch.no_grad():
        for b in range(B):
            boxes = boxes_batch[b]
            if len(boxes) == 0:
                final_predictions.append(torch.zeros((1, 1, out_size, out_size), device=device))
                continue
            curr_embed = image_embeddings[b:b+1]
            boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)  # [N,4] xyxy
            # clip to image bounds (half-open interval)
            boxes_t = boxes_t.clamp_(min=0.0, max=float(out_size))

            sparse, dense = sam_model.prompt_encoder(points=None, boxes=boxes_t, masks=None)
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=curr_embed,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            up = F.interpolate(low_res_masks, size=(out_size, out_size), mode='bilinear', align_corners=False)
            merged = torch.max(up, dim=0)[0]  # [1,H,W]
            final_predictions.append(merged.unsqueeze(0))  # [1,1,H,W]
    return torch.cat(final_predictions, dim=0)

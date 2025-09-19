# lora.py
from segment_anything.modeling.sam import Sam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file


class LoRA_qkv(nn.Module):
    """
    LoRA adaptation for attention modules (queries and values only).

    This wraps an existing linear `qkv` projection with two low-rank adapters:
    one for queries (q) and one for values (v). Keys are left untouched.
    """
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, dropout_p=0.1):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor):
        # Original qkv pass-through
        qkv = self.qkv(x)
        # LoRA branches for q and v
        q_ba = self.linear_b_q(self.dropout(self.linear_a_q(x)))
        v_ba = self.linear_b_v(self.dropout(self.linear_a_v(x)))
        # Fuse back into qkv: shape [B, HW, 3*D] before attention reshape in SAM
        qkv[:, :, :, :self.d_model]  += q_ba  # q
        qkv[:, :, :, -self.d_model:] += v_ba  # v
        return qkv


class LoRA_sam(nn.Module):
    """
    Wrap the SAM image encoder with LoRA adapters on selected blocks.

    - Keep the SAM backbone frozen; only the LoRA parameters are trainable.
    - A/B matrices are tracked for saving/loading with safetensors.
    """
    def __init__(self, sam_model: Sam, rank: int, lora_layer=None, dropout_p=0.1):
        super().__init__()
        assert rank > 0
        self.rank = rank
        self.dropout_p = dropout_p

        # Select which encoder blocks receive LoRA
        if lora_layer is not None:
            self.lora_layer = list(lora_layer)
            print(f"LoRA on layers: {self.lora_layer}")
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        # Freeze the image encoder
        for p in sam_model.image_encoder.parameters():
            p.requires_grad = False

        # Track LoRA linear layers for saving/loading
        # Python lists are kept for compatibility with existing training scripts
        self.A_weights = []
        self.B_weights = []

        # Insert LoRA on selected layers
        for layer_idx, blk in enumerate(sam_model.image_encoder.blocks):
            if layer_idx not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            d_model = w_qkv_linear.in_features
            dev = w_qkv_linear.weight.device  # keep on the same device as original qkv

            # Low-rank factors A/B (no bias)
            w_a_linear_q = nn.Linear(d_model, rank, bias=False).to(dev)
            w_b_linear_q = nn.Linear(rank,   d_model, bias=False).to(dev)
            w_a_linear_v = nn.Linear(d_model, rank, bias=False).to(dev)
            w_b_linear_v = nn.Linear(rank,   d_model, bias=False).to(dev)

            # Register for I/O and checkpointing
            self.A_weights.extend([w_a_linear_q, w_a_linear_v])
            self.B_weights.extend([w_b_linear_q, w_b_linear_v])

            # Replace original qkv with LoRA-augmented module
            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q, w_b_linear_q,
                w_a_linear_v, w_b_linear_v,
                dropout_p=self.dropout_p
            )

        self.reset_parameters()

        # Register SAM as a submodule so .to()/.eval() cascade properly
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder  # backward-compat alias

    def reset_parameters(self):
        """Initialize LoRA factors: A with Kaiming uniform, B with zeros."""
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str):
        """Save LoRA weights as safetensors using sequential keys w_a_000, w_b_000, ..."""
        a_tensors = {f"w_a_{i:03d}": lin.weight.detach().cpu() for i, lin in enumerate(self.A_weights)}
        b_tensors = {f"w_b_{i:03d}": lin.weight.detach().cpu() for i, lin in enumerate(self.B_weights)}
        merged = {**a_tensors, **b_tensors}
        save_file(merged, filename)

    def load_lora_parameters(self, filename: str):
        """
        Load LoRA weights from safetensors and copy them onto the device of `self.sam`.

        Uses in-place copy_ to avoid replacing Parameters (prevents optimizer references
        from being invalidated).
        """
        dev = next(self.sam.parameters()).device
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                key = f"w_a_{i:03d}"
                t = f.get_tensor(key).to(dev)
                w_A_linear.weight.data.copy_(t)

            for i, w_B_linear in enumerate(self.B_weights):
                key = f"w_b_{i:03d}"
                t = f.get_tensor(key).to(dev)
                w_B_linear.weight.data.copy_(t)

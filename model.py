import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP with (num_layers-1) hidden ReLU blocks followed by a linear head."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        h = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            h += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        h += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*h)

    def forward(self, x):
        return self.net(x)


class PositionEmbeddingSine(nn.Module):
    """Standard 2D sine-cosine positional encoding producing [B, C, H, W]."""
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or 2 * math.pi

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = self.temperature ** (2 * (torch.arange(self.num_pos_feats, device=device) // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)
        return pos.unsqueeze(0).repeat(B, 1, 1, 1)


class Prompt_Predictor(nn.Module):
    def __init__(self, num_classes=1, num_queries=100, hidden_dim=256,
                 nheads=8, dec_layers=6, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.input_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # + no-object
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, num_layers=3)  # cxcywh in [0, 1]

    def forward(self, image_embedding, inter_features=None):
        # Project channels and add 2D sine-cosine positional encoding
        src = self.input_proj(image_embedding)               # [B, C, H, W]
        pos = self.position_embedding(src)

        # Flatten spatial dims to a sequence
        B, C, H, W = src.shape
        src = src.flatten(2).permute(0, 2, 1)                # [B, HW, C]
        pos = pos.flatten(2).permute(0, 2, 1)                # [B, HW, C]

        memory = src + pos
        tgt = torch.zeros(B, self.num_queries, self.hidden_dim, device=src.device)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, C]

        hs = self.decoder(tgt + query_pos, memory)           # [B, Q, C]
        outputs_class = self.class_embed(hs)                  # [B, Q, K+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()         # [B, Q, 4]
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

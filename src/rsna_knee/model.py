"""Stage A (2.5D per-series encoder) + Stage B (cross-slot study head) per CLAUDE.md §5.

Stage A: DINOv2 ViT-S/14 backbone applied per-slice, then a small transformer over the T slice
tokens (learned CLS token + positional encoding by normalized slice index) -> one series embedding.

Stage B: learned slot identity embeddings + a cross-slot transformer over the <=6 series
embeddings, then per-target attention pooling (12 learned query vectors, one per target, each
attends over the slot embeddings) -> 12 logits.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

NUM_SLOTS = 6


class SlotModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_targets: int = 12,
        slice_layers: int = 2,
        slice_heads: int = 4,
        slot_layers: int = 2,
        slot_heads: int = 4,
    ):
        super().__init__()
        # dynamic_img_size=True: DINOv2 timm configs default to a fixed 518x518 img_size and
        # assert on it otherwise. Our cache is 384px (resized to the nearest patch-14 multiple,
        # see dataset._nearest_patch_multiple) — this interpolates the pretrained position
        # embeddings to whatever size we actually feed it instead of hardcoding a resize to 518.
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, in_chans=3, dynamic_img_size=True
        )
        D = self.backbone.num_features
        self.embed_dim = D

        self.slice_pos_embed = nn.Sequential(nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.slice_cls = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.slice_cls, std=0.02)
        slice_layer = nn.TransformerEncoderLayer(D, slice_heads, dim_feedforward=D * 4, batch_first=True)
        self.slice_transformer = nn.TransformerEncoder(slice_layer, slice_layers)

        self.slot_embed = nn.Embedding(NUM_SLOTS, D)
        slot_layer = nn.TransformerEncoderLayer(D, slot_heads, dim_feedforward=D * 4, batch_first=True)
        self.slot_transformer = nn.TransformerEncoder(slot_layer, slot_layers)

        self.target_queries = nn.Parameter(torch.randn(num_targets, D) * 0.02)
        self.target_attn = nn.MultiheadAttention(D, slot_heads, batch_first=True)
        self.head = nn.Linear(D, 1)

    def encode_series(self, volumes: torch.Tensor) -> torch.Tensor:
        """volumes: [B, S, T, H, W] single-channel -> series embeddings [B, S, D]."""
        B, S, T, H, W = volumes.shape
        x = volumes.reshape(B * S * T, 1, H, W).repeat(1, 3, 1, 1)
        feats = self.backbone(x)  # [B*S*T, D]
        D = feats.shape[-1]
        feats = feats.view(B * S, T, D)

        pos = torch.linspace(0, 1, T, device=feats.device).view(1, T, 1).expand(B * S, T, 1)
        feats = feats + self.slice_pos_embed(pos)

        cls = self.slice_cls.expand(B * S, 1, D)
        seq = torch.cat([cls, feats], dim=1)
        seq = self.slice_transformer(seq)
        return seq[:, 0].view(B, S, D)

    def forward(self, volumes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """volumes: [B,6,T,H,W], mask: [B,6] bool (True=present) -> logits [B, num_targets]."""
        series_embed = self.encode_series(volumes)  # [B,6,D]
        series_embed = series_embed + self.slot_embed.weight.unsqueeze(0)

        key_padding_mask = ~mask  # True = ignore
        slot_ctx = self.slot_transformer(series_embed, src_key_padding_mask=key_padding_mask)

        B = volumes.shape[0]
        q = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.target_attn(q, slot_ctx, slot_ctx, key_padding_mask=key_padding_mask)
        return self.head(pooled).squeeze(-1)


def get_param_groups(model: SlotModel, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else other_params).append(p)
    return [
        {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]

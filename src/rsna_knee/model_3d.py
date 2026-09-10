# Copyright 2026 Husnain Rasool
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D architecture family (exp007, EXECUTION_PLAN Phase 3.4) — ensemble-diversity candidate, not a
solo-performance bid. Stage A: MONAI 3D ResNet-50 applied to a whole [T,H,W] volume per series at
once (replaces the 2D DINOv2 ViT + slice-attention-transformer encoder entirely — a native 3D conv
backbone sees inter-slice spatial structure directly instead of aggregating independently-encoded
2D slices). Stage B (study head): a separate implementation from model.py's SlotModel — deliberately
NOT sharing a base class, to keep zero risk of touching the working 2D path (exp001-006) — but the
architecture and hyperparameters are identical (slot embeddings, 2-layer pre-norm cross-slot
transformer, per-target attention pooling, same small-gain init).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from monai.networks.nets import resnet50

from .model import SMALL_GAIN_STD, _small_gain_init_attn, _small_gain_init_encoder

NUM_SLOTS = 6


class Slot3DModel(nn.Module):
    def __init__(self, num_targets: int = 12, slot_layers: int = 2, slot_heads: int = 4):
        super().__init__()
        # feed_forward=False, num_classes=0: raw 2048-dim pooled feature, no classification head —
        # same "backbone as pure feature extractor" pattern as SlotModel's timm ViT (num_classes=0).
        self.backbone = resnet50(spatial_dims=3, n_input_channels=1, num_classes=0, feed_forward=False)
        self.embed_dim = 2048
        D = self.embed_dim

        self.slot_embed = nn.Embedding(NUM_SLOTS, D)
        slot_layer = nn.TransformerEncoderLayer(D, slot_heads, dim_feedforward=D * 4,
                                                 batch_first=True, norm_first=True)
        self.slot_transformer = nn.TransformerEncoder(slot_layer, slot_layers)
        _small_gain_init_encoder(self.slot_transformer)

        self.target_queries = nn.Parameter(torch.randn(num_targets, D) * SMALL_GAIN_STD)
        self.target_attn = nn.MultiheadAttention(D, slot_heads, batch_first=True)
        _small_gain_init_attn(self.target_attn)
        self.head = nn.Linear(D, 1)

    def encode_series(self, volumes: torch.Tensor, mask: torch.Tensor, chunk_size: int = 2) -> torch.Tensor:
        """volumes: [B,S,T,H,W] single-channel, mask: [B,S] bool -> series embeddings [B,S,D].

        chunk_size is far smaller than the 2D path's (128) — a 3D-conv forward over a whole
        T x H x W volume is dramatically more expensive per item than one 2D slice, so peak memory
        is bounded by processing very few volumes at a time, backed by gradient checkpointing
        (torch.utils.checkpoint wraps each chunk's backbone call) so activations are recomputed on
        the backward pass instead of held for the whole batch."""
        B, S, T, H, W = volumes.shape
        D = self.embed_dim
        device = volumes.device

        flat_mask = mask.reshape(B * S)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]

        series_embed = torch.zeros(B * S, D, device=device, dtype=volumes.dtype)
        if valid_idx.numel() == 0:
            return series_embed.view(B, S, D)

        valid_volumes = volumes.reshape(B * S, T, H, W)[valid_idx]  # [N_valid, T, H, W]
        x = valid_volumes.unsqueeze(1)  # [N_valid, 1, T, H, W] — single-channel 3D volume

        feats_chunks = []
        for start in range(0, x.shape[0], chunk_size):
            chunk = x[start:start + chunk_size]
            feats_chunks.append(torch_checkpoint.checkpoint(self.backbone, chunk, use_reentrant=False))
        feats = torch.cat(feats_chunks, dim=0)  # [N_valid, D]
        series_embed[valid_idx] = feats.to(series_embed.dtype)
        return series_embed.view(B, S, D)

    def forward(self, volumes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """volumes: [B,6,T,H,W], mask: [B,6] bool (True=present) -> logits [B, num_targets]."""
        series_embed = self.encode_series(volumes, mask)
        series_embed = series_embed + self.slot_embed.weight.unsqueeze(0)

        key_padding_mask = ~mask
        slot_ctx = self.slot_transformer(series_embed, src_key_padding_mask=key_padding_mask)

        B = volumes.shape[0]
        q = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.target_attn(q, slot_ctx, slot_ctx, key_padding_mask=key_padding_mask)
        return self.head(pooled).squeeze(-1)


def get_param_groups(model: Slot3DModel, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else other_params).append(p)
    return [
        {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]

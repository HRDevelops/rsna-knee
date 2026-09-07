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
SMALL_GAIN_STD = 0.02


def _small_gain_init_attn(attn: nn.MultiheadAttention, std: float = SMALL_GAIN_STD) -> None:
    """Re-init an nn.MultiheadAttention's Q/K/V and output projections with a small std, instead
    of PyTorch's default Xavier-uniform init (which is scaled for a module trained from a
    well-conditioned start, not for a from-scratch attention block sitting downstream of a
    pretrained ViT backbone with very different activation statistics)."""
    if attn.in_proj_weight is not None:
        nn.init.trunc_normal_(attn.in_proj_weight, std=std)
    if attn.in_proj_bias is not None:
        nn.init.zeros_(attn.in_proj_bias)
    nn.init.trunc_normal_(attn.out_proj.weight, std=std)
    if attn.out_proj.bias is not None:
        nn.init.zeros_(attn.out_proj.bias)


def _small_gain_init_encoder(encoder: nn.TransformerEncoder, std: float = SMALL_GAIN_STD) -> None:
    for layer in encoder.layers:
        _small_gain_init_attn(layer.self_attn, std=std)


class SlotModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_targets: int = 12,
        slice_layers: int = 2,
        slice_heads: int = 4,
        slot_layers: int = 2,
        slot_heads: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        # dynamic_img_size=True: DINOv2 timm configs default to a fixed 518x518 img_size and
        # assert on it otherwise. Our cache is 384px (resized to the nearest patch-14 multiple,
        # see dataset._nearest_patch_multiple) — this interpolates the pretrained position
        # embeddings to whatever size we actually feed it instead of hardcoding a resize to 518.
        # pretrained=False at inference time (Kaggle notebooks, internet OFF): we load our own
        # fine-tuned state_dict immediately after construction anyway, so the ImageNet/DINOv2 hub
        # download `pretrained=True` would trigger is both unreachable and unnecessary there.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, in_chans=3, dynamic_img_size=True
        )
        # Chunking the backbone calls (see encode_series) does NOT reduce training memory on its
        # own — autograd keeps every chunk's activations alive for backward regardless of how the
        # forward is split (confirmed by OOM persisting after adding chunking). Gradient
        # checkpointing actually discards and recomputes activations, which is what's needed for
        # a B*S*T up to ~1152-image effective batch through a ViT.
        self.backbone.set_grad_checkpointing(enable=True)
        D = self.backbone.num_features
        self.embed_dim = D

        self.slice_pos_embed = nn.Sequential(nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.slice_cls = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.slice_cls, std=SMALL_GAIN_STD)
        # norm_first=True (pre-norm): stability bundle item — from-scratch post-norm transformer
        # blocks sitting downstream of a pretrained ViT are prone to activation blow-up early in
        # training; pre-norm keeps the residual stream well-behaved from step 0.
        slice_layer = nn.TransformerEncoderLayer(D, slice_heads, dim_feedforward=D * 4,
                                                  batch_first=True, norm_first=True)
        self.slice_transformer = nn.TransformerEncoder(slice_layer, slice_layers)
        _small_gain_init_encoder(self.slice_transformer)

        self.slot_embed = nn.Embedding(NUM_SLOTS, D)
        slot_layer = nn.TransformerEncoderLayer(D, slot_heads, dim_feedforward=D * 4,
                                                 batch_first=True, norm_first=True)
        self.slot_transformer = nn.TransformerEncoder(slot_layer, slot_layers)
        _small_gain_init_encoder(self.slot_transformer)

        self.target_queries = nn.Parameter(torch.randn(num_targets, D) * SMALL_GAIN_STD)
        self.target_attn = nn.MultiheadAttention(D, slot_heads, batch_first=True)
        _small_gain_init_attn(self.target_attn)
        self.head = nn.Linear(D, 1)

    def encode_series(self, volumes: torch.Tensor, mask: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """volumes: [B, S, T, H, W] single-channel, mask: [B,S] bool -> series embeddings [B, S, D].

        A naive B*S*T forward (up to ~8*6*24=1152 images at once) OOMs an 80GB H100 outright —
        fixed two ways: (1) absent slots (mask=False) are skipped entirely rather than wastefully
        encoding zero-tensors, since most studies don't fill all 6 slots; (2) the backbone is
        called in chunks to bound peak activation memory regardless of how many valid slots exist.
        """
        B, S, T, H, W = volumes.shape
        D = self.embed_dim
        device = volumes.device

        flat_mask = mask.reshape(B * S)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]

        series_embed = torch.zeros(B * S, D, device=device, dtype=volumes.dtype)
        if valid_idx.numel() == 0:
            return series_embed.view(B, S, D)

        valid_volumes = volumes.reshape(B * S, T, H, W)[valid_idx]  # [N_valid, T, H, W]
        n_valid = valid_volumes.shape[0]
        x = valid_volumes.reshape(n_valid * T, 1, H, W).repeat(1, 3, 1, 1)

        feats_chunks = [self.backbone(x[start:start + chunk_size]) for start in range(0, x.shape[0], chunk_size)]
        feats = torch.cat(feats_chunks, dim=0).view(n_valid, T, D)

        pos = torch.linspace(0, 1, T, device=device).view(1, T, 1).expand(n_valid, T, 1)
        feats = feats + self.slice_pos_embed(pos)

        cls = self.slice_cls.expand(n_valid, 1, D)
        seq = torch.cat([cls, feats], dim=1)
        seq = self.slice_transformer(seq)
        series_embed[valid_idx] = seq[:, 0].to(series_embed.dtype)
        return series_embed.view(B, S, D)

    def forward(self, volumes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """volumes: [B,6,T,H,W], mask: [B,6] bool (True=present) -> logits [B, num_targets]."""
        series_embed = self.encode_series(volumes, mask)  # [B,6,D]
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

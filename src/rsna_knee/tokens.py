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

"""Token transformer of docs/research/PLAN_v1_3.md s2.4-2.6 without geometry = B_cap (Phase 0C). The geometry model G
(Phase 1) adds a cross-plane footprint bias to `attn_bias` and nothing else, so B_cap is G at beta = 0.

Per window: CoAtNet-2 final map (12x12 at 384 px; 336 px input is resized to 384 as in exp018) -> adaptive average
pool to g x g -> tokens. Each token gets: projected feature + plane + series-slot + 2D grid + slice-position (sinusoidal)
+ acquisition-metadata embeddings. L pre-norm transformer layers over all tokens of one pass, with a learned per-head
same-series bias b_h and key masking of tokens that are mostly padding (validity < 0.5). Readout: exp018's per-finding
attention-MIL over the valid tokens of the pass. Bagged inference averages the sigmoid of the per-pass logits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.wmil import prepare_input
from rsna_knee.wmil_v2 import N_META


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):                       # x [B, N, d], bias [B, H, N, N] (additive, -inf = masked)
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, d)))
        return x + self.drop(self.mlp(self.n2(x)))


class TokenMIL(nn.Module):
    def __init__(self, arch="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", grid=6, d=384, heads=8, layers=2,
                 n_targets=12, drop=0.2, pretrained=True, grad_checkpointing=True, geometry="none", eps=1.0,
                 features="final"):
        super().__init__()
        assert geometry in ("none", "cross_plane") and features in ("final", "local")
        self.features = features
        self.geometry, self.eps = geometry, eps
        # beta_h >= 0 per layer and head, EXACTLY 0 at init (PHASE1_PREREG s2); projected SGD in the trainer.
        # Created before any randomly initialized module and without RNG use, so B_cap and G share every init draw.
        self.beta = nn.Parameter(torch.zeros(layers, heads)) if geometry != "none" else None
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="")
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        c = self.backbone.num_features
        if features == "local":
            # PHASE2_LOCAL_PREREG: output of backbone.stages.1 (last MBConv stage, stride 8, 256 ch), i.e. before the
            # first global-attention stage (stages.2, TransformerBlock2d). Stages 2-3 and the final norm are removed.
            self.backbone.stages = self.backbone.stages[:2]
            self.backbone.norm = nn.Identity()
            c = self.backbone.stages[1].blocks[-1].conv3_1x1.out_channels if hasattr(
                self.backbone.stages[1].blocks[-1], "conv3_1x1") else 256
        self.grid, self.heads = grid, heads
        self.proj = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, d))
        self.plane_emb = nn.Embedding(3, d)
        self.slot_emb = nn.Embedding(16, d)
        self.grid_emb = nn.Parameter(torch.zeros(grid * grid, d))
        self.pos_freq = nn.Parameter(torch.tensor([2 ** k * math.pi for k in range(8)]), requires_grad=False)
        self.pos_proj = nn.Linear(16, d)
        self.meta = nn.Sequential(nn.Linear(N_META, d), nn.GELU(), nn.Linear(d, d))
        self.same_series = nn.Parameter(torch.zeros(heads))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_targets))
        self.cls_w = nn.Parameter(torch.zeros(n_targets, d))
        self.cls_b = nn.Parameter(torch.zeros(n_targets))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        nn.init.trunc_normal_(self.grid_emb, std=0.02)

    def encode(self, windows_u8, res, gain=None, chunk=96):
        """[B, K, 3, H, W] uint8 -> [B, K*g*g, C] token features (chunked through the backbone)."""
        B, K = windows_u8.shape[:2]
        flat = windows_u8.flatten(0, 1)
        g = None if gain is None else gain.repeat_interleave(K)   # one gain per study, as exp018
        outs = []
        for s in range(0, flat.shape[0], chunk):
            # prepare_input applies one gain per leading index: give each window its own leading index
            part = flat[s:s + chunk].unsqueeze(1)                     # [n, 1, 3, H, W]
            x = prepare_input(part, res, None if g is None else g[s:s + chunk])
            f = self.backbone.forward_features(x)
            outs.append(F.adaptive_avg_pool2d(f, self.grid).flatten(2).transpose(1, 2))   # [n, g*g, C]
        return torch.cat(outs).view(B, K * self.grid * self.grid, -1)

    def footprint_d2(self, mu, sigma, plane, chunk=256):
        """Squared Mahalanobis footprint distance between all token pairs (fp32) and the cross-plane mask.
        mu [B, K, g, g, 3]; sigma [B, K, 3, 3] (one covariance per window); plane [B, K]."""
        B, K = mu.shape[:2]
        gg = self.grid * self.grid
        m = mu.float().reshape(B, K * gg, 3)
        S = sigma.float()
        Minv = torch.linalg.inv(S[:, :, None] + S[:, None, :] + self.eps * torch.eye(3, device=mu.device))  # [B,K,K,3,3]
        win = torch.arange(K, device=mu.device).repeat_interleave(gg)
        N = K * gg
        d2 = torch.empty(B, N, N, device=mu.device)
        for s in range(0, N, chunk):
            D = m[:, s:s + chunk, None, :] - m[:, None, :, :]                       # [B, c, N, 3]
            Mi = Minv[:, win[s:s + chunk]][:, :, win]                              # [B, c, N, 3, 3]
            d2[:, s:s + chunk] = torch.einsum("bcnx,bcnxy,bcny->bcn", D, Mi, D)
        pt = plane.repeat_interleave(gg, dim=1)
        cross = (pt[:, :, None] != pt[:, None, :])
        return d2, cross

    def forward(self, feats, slot, plane, pos, meta, valid, extra_bias=None, mu=None, sigma=None):
        """feats [B, N, C] (N = K*g*g); per-window slot/plane/pos/meta [B, K(, M)]; valid [B, K, g, g]."""
        B, N, _ = feats.shape
        gg = self.grid * self.grid
        K = N // gg
        rep = lambda t: t.repeat_interleave(gg, dim=1)                                   # noqa: E731
        ang = rep(pos)[..., None] * self.pos_freq
        x = self.proj(feats) + self.plane_emb(rep(plane)) + self.slot_emb(rep(slot)) \
            + self.grid_emb.repeat(K, 1)[None] + self.pos_proj(torch.cat([ang.sin(), ang.cos()], -1)) \
            + self.meta(rep(meta))
        keep = valid.flatten(1) >= 0.5                                                    # [B, N]
        same = (rep(slot)[:, :, None] == rep(slot)[:, None, :]).to(x.dtype)               # [B, N, N]
        bias = same[:, None] * self.same_series.view(1, -1, 1, 1).to(x.dtype)            # [B, H, N, N]
        if extra_bias is not None:
            bias = bias + extra_bias
        bias = bias.masked_fill(~keep[:, None, None, :], float("-inf"))
        geo = None
        if self.geometry == "cross_plane":
            d2, cross = self.footprint_d2(mu, sigma, plane)
            geo = (d2 * cross)[:, None]                                              # [B, 1, N, N], fp32
        for li, blk in enumerate(self.blocks):
            if geo is None:
                x = blk(x, bias)
            else:
                x = blk(x, bias.float() - 0.5 * self.beta[li].view(1, -1, 1, 1) * geo)
        h = self.norm(x)
        a = self.att(h).masked_fill(~keep[..., None], -1e4)
        a = torch.softmax(a.float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bnt,bnf->btf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    timesteps: [B]
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int, emb_ch: int, out_ch: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_ch = out_ch or ch
        self.in_ch = ch
        self.out_ch = out_ch
        self.norm1 = GroupNorm32(32, ch)
        self.conv1 = nn.Conv2d(ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.norm2 = GroupNorm32(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if ch == out_ch else nn.Conv2d(ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 1):
        super().__init__()
        self.norm = GroupNorm32(32, ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        
        # For ADM with model_channels=192, typical attention heads:
        # - At 192 channels: 3 heads (64 per head)
        # - At 384 channels: 6 heads (64 per head)
        # - At 768 channels: 12 heads (64 per head)
        # Ensure channel dimension is divisible by num_heads
        # We'll use a fixed mapping based on channel size
        if ch >= 768:
            self.num_heads = 12
        elif ch >= 384:
            self.num_heads = 6
        elif ch >= 192:
            self.num_heads = 3
        else:
            self.num_heads = 1
            
        # Double-check divisibility
        while ch % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, h * w)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # reshape heads
        head_dim = c // self.num_heads
        # Reshape to (batch, num_heads, head_dim, seq_len)
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        scale = 1.0 / math.sqrt(head_dim)  # Use head_dim directly, not sqrt(sqrt(head_dim))
        weight = torch.einsum("bnhd,bnhd->bnh", q * scale, k * scale)  # Compute attention scores
        weight = torch.softmax(weight, dim=-1)
        out = torch.einsum("bnh,bnhd->bnhd", weight, v)  # Apply attention weights
        out = out.reshape(b, c, h * w)  # Reshape back to (batch, channels, seq_len)
        out = self.proj(out).view(b, c, h, w)  # Project and reshape to original dimensions
        return x_in + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


@dataclass
class ModelConfig:
    image_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 192  # Updated to match checkpoint (was 128)
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_heads: int = 1  # Will be calculated as model_channels // 192 in AttentionBlock
    dropout: float = 0.0
    time_embed_dim: int = 768  # Updated to match checkpoint (was 512)


class UNetModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(ch, cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )

        self.input_conv = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        hs_c = [ch]
        cur_res = cfg.image_size
        cur_ch = ch
        for level, mult in enumerate(cfg.channel_mult):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(cur_ch, cfg.time_embed_dim, out_ch, dropout=cfg.dropout))
                cur_ch = out_ch
                if cur_res in cfg.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(cur_ch, num_heads=cfg.num_heads))
                hs_c.append(cur_ch)
            if level != len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(cur_ch))
                cur_res //= 2
                hs_c.append(cur_ch)

        # Middle
        self.mid = nn.ModuleList([
            ResBlock(cur_ch, cfg.time_embed_dim, cur_ch, dropout=cfg.dropout),
            AttentionBlock(cur_ch, num_heads=cfg.num_heads),
            ResBlock(cur_ch, cfg.time_embed_dim, cur_ch, dropout=cfg.dropout),
        ])

        # Up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = hs_c.pop()
                self.up_blocks.append(ResBlock(cur_ch + skip_ch, cfg.time_embed_dim, out_ch, dropout=cfg.dropout))
                cur_ch = out_ch
                if cur_res in cfg.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(cur_ch, num_heads=cfg.num_heads))
            if level != 0:
                self.upsamples.append(Upsample(cur_ch))
                cur_res *= 2

        self.out_norm = GroupNorm32(32, cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # x in [-1,1]; output epsilon prediction
        emb = timestep_embedding(timesteps, self.cfg.model_channels)
        emb = self.time_embed(emb)

        hs = []
        h = self.input_conv(x)
        hs.append(h)

        di = 0
        ds = 0
        cur_res = self.cfg.image_size
        # down blocks
        for level, mult in enumerate(self.cfg.channel_mult):
            for _ in range(self.cfg.num_res_blocks):
                # ResBlock
                h = self.down_blocks[di](h, emb); di += 1
                # optional attn
                if isinstance(self.down_blocks[di], AttentionBlock):
                    h = self.down_blocks[di](h); di += 1
                hs.append(h)
            if level != len(self.cfg.channel_mult) - 1:
                h = self.downsamples[ds](h); ds += 1
                cur_res //= 2
                hs.append(h)

        # mid
        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)

        # up
        ui = 0
        us = 0
        for level, mult in reversed(list(enumerate(self.cfg.channel_mult))):
            for _ in range(self.cfg.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[ui](h, emb); ui += 1
                if ui < len(self.up_blocks) and isinstance(self.up_blocks[ui], AttentionBlock):
                    h = self.up_blocks[ui](h); ui += 1
            if level != 0:
                h = self.upsamples[us](h); us += 1
                cur_res *= 2

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h

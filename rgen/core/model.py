# rgen/core/model.py
from __future__ import annotations
import math
from typing import Iterable, List, Optional
import torch
from torch import nn

# ---- Embeddings ----
class SinusoidalPosEmb(nn.Module):
    """Standard 1D sinusoidal embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        # exp(-linspace(log(1), log(10000), half)) gives 1/exp(...)
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device) * (-1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

class NewSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Standard DDPM/ADM embedding.
        t: (B,) int or float timesteps in [0, T)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=device).float() / half
        )  # [half] in (1, 1/max_period]
        args = t.float()[:, None] * freqs[None, :]  # (B,half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb

def make_mlp(in_dim: int, out_dim: int, hidden: Optional[int] = None) -> nn.Sequential:
    h = hidden or out_dim * 4
    return nn.Sequential(
        nn.Linear(in_dim, h), nn.SiLU(),
        nn.Linear(h, out_dim),
    )

# ---- Blocks ----

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2  = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hds = self.num_heads
        x_ = self.norm(x)
        qkv = self.qkv(x_)  # (b, 3c, h, w)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, hds, c // hds, h * w)
        k = k.reshape(b, hds, c // hds, h * w)
        v = v.reshape(b, hds, c // hds, h * w)
        attn = torch.softmax((q.transpose(-2, -1) @ k) / math.sqrt(c // hds), dim=-1)  # (b,h,HW,HW)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)                           # (b,h,HW,C//h)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x): return self.op(x)

# ---- U-Net ----

class UNet32(nn.Module):
    """
    U-Net for 32x32 images. Class-conditional with null-label support (for CFG later).
    Skip stack is pushed:
      - after input conv,
      - after each ResBlock,
      - after each Downsample.
    On the up path, we concat one skip per ResBlock (num_res_blocks + 1 per level).
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        ch_mults: Iterable[int] = (1, 2, 2),
        num_res_blocks: int = 3,
        attn_resolutions: Iterable[int] = (16,),
        num_classes: int = 10,
        cond_null_id: int = 10,
        dropout: float = 0.0,
        predict_sigma: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch * (2 if predict_sigma else 1)
        self.predict_sigma = predict_sigma
        self.num_classes = num_classes
        self.cond_null_id = cond_null_id

        time_dim = base_ch * 4
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Label embedding (includes null id at index num_classes)
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

        # Input conv
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---- Down path (record skip channels only after ResBlocks and Downsamples)
        downs = []
        skip_chs: List[int] = [base_ch]  # after in_conv
        curr_res = 32
        in_chs = base_ch
        for i, mult in enumerate(ch_mults):
            out_chs = base_ch * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock(in_chs, out_chs, time_dim, dropout))
                in_chs = out_chs
                if curr_res in attn_resolutions:
                    downs.append(SelfAttention2D(in_chs))
                skip_chs.append(in_chs)  # after ResBlock (not after attention)
            if i != len(ch_mults) - 1:
                downs.append(Downsample(in_chs))
                curr_res //= 2
                skip_chs.append(in_chs)  # after Downsample

        self.down = nn.ModuleList(downs)

        # Middle
        self.mid1 = ResBlock(in_chs, in_chs, time_dim, dropout)
        self.mid_attn = SelfAttention2D(in_chs)
        self.mid2 = ResBlock(in_chs, in_chs, time_dim, dropout)

        # ---- Up path
        # Build using a copy of skip_chs so we know exact concat sizes at construction time.
        ups = []
        build_skips = skip_chs.copy()
        for i, mult in list(enumerate(ch_mults))[::-1]:
            out_chs = base_ch * mult
            # num_res_blocks + 1 cats per level (last one uses the pre-downsample skip)
            for _ in range(num_res_blocks + 1):
                skip_in = build_skips.pop()
                ups.append(ResBlock(in_chs + skip_in, out_chs, time_dim, dropout))
                in_chs = out_chs
                if curr_res in attn_resolutions:
                    ups.append(SelfAttention2D(in_chs))
            if i != 0:
                ups.append(Upsample(in_chs))
                curr_res *= 2
        self.up = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(32, in_chs)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_chs, self.out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,3,32,32) in [-1,1]
        t: (B,) int64 timesteps
        y: (B,) int64 labels in [0..num_classes] where `cond_null_id` is the null.
        """
        temb = self.time_emb(t)
        if y is None:
            y = torch.full_like(t, self.cond_null_id)
        y = torch.clamp(y, 0, self.num_classes)  # allow null id == num_classes
        temb = temb + self.label_emb(y)

        h = self.in_conv(x)
        hs: List[torch.Tensor] = [h]

        # Down: push to hs after ResBlock and after Downsample only
        for m in self.down:
            if isinstance(m, ResBlock):
                h = m(h, temb)
                hs.append(h)
            elif isinstance(m, Downsample):
                h = m(h)
                hs.append(h)
            else:  # SelfAttention2D
                h = m(h)

        # Middle
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        # Up: consume one skip per ResBlock
        for m in self.up:
            if isinstance(m, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = m(h, temb)
            else:
                h = m(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

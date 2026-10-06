"""
ConvNeXt V2 Backbone
العمود الفقري ConvNeXt V2

Reference:
    Woo et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .attention import get_attention_module


class GRN(nn.Module):
    """
    Global Response Normalization Layer

    Normalizes features based on global context for better feature diversity.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) regularization.

    Randomly drops entire residual branches during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 Block

    Features:
    - Depthwise convolution (7x7)
    - LayerNorm
    - Inverted bottleneck (expand 4x)
    - GELU activation
    - Global Response Normalization (GRN)
    - Drop Path regularization
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Normalization (applied in channel-last format)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted bottleneck
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Channel-last format for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        # Inverted bottleneck with GRN
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        # Back to channel-first format
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Residual connection with drop path
        x = residual + self.drop_path(x)

        return x


class ConvNeXtV2Stage(nn.Module):
    """
    ConvNeXt V2 Stage

    Contains multiple ConvNeXt V2 blocks with optional attention.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: List[float],
        use_attention: bool = True,
        attention_type: str = 'cbam'
    ):
        super().__init__()

        # ConvNeXt V2 blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtV2Block(dim, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

        # Optional attention
        if use_attention:
            self.attention = get_attention_module(attention_type, dim)
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.attention(x)
        return x


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXt V2 Backbone with Attention

    Multi-scale feature extraction backbone based on ConvNeXt V2 architecture
    with integrated attention mechanisms.

    Architecture:
        Stem (4x downsample) -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4

    Features:
        - Global Response Normalization (GRN)
        - Stochastic Depth (Drop Path)
        - Configurable attention (CBAM, SE, ECA)
        - Multi-scale feature output
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        depths: List[int] = None,
        drop_path_rate: float = 0.1,
        use_attention: bool = True,
        attention_type: str = 'cbam'
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]

        assert len(depths) == num_stages, "depths must match num_stages"

        self.num_stages = num_stages

        # Channel dimensions for each stage
        dims = [base_channels * (2 ** i) for i in range(num_stages)]
        self.out_channels = dims

        # Stem: 4x downsample
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )

        # Compute drop path rates for each block
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur = 0
        for i in range(num_stages):
            # Stage blocks
            stage = ConvNeXtV2Stage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur:cur + depths[i]],
                use_attention=use_attention,
                attention_type=attention_type
            )
            self.stages.append(stage)
            cur += depths[i]

            # Downsample layer (except for last stage)
            if i < num_stages - 1:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature tensors at each stage
        """
        features = []

        # Stem
        x = self.stem(x)

        # Stages
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)

            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        return features

    def get_feature_info(self) -> List[dict]:
        """Get information about output features at each stage."""
        info = []
        stride = 4  # After stem

        for i, channels in enumerate(self.out_channels):
            info.append({
                'stage': i + 1,
                'channels': channels,
                'stride': stride
            })
            stride *= 2

        return info


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (channel-first format)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

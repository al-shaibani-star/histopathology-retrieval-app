"""
Attention Mechanisms
آليات الانتباه

Includes:
- CBAM (Convolutional Block Attention Module)
- SE (Squeeze-and-Excitation)
- ECA (Efficient Channel Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """
    Channel Attention Module

    Uses both average and max pooling to capture channel-wise statistics.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    Uses channel-wise pooling to focus on important spatial regions.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)

    Sequential application of channel and spatial attention.

    Reference:
        Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    Adaptively recalibrates channel-wise feature responses.

    Reference:
        Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block

    Uses 1D convolution for efficient channel attention without dimensionality reduction.

    Reference:
        Wang et al. "ECA-Net: Efficient Channel Attention" (CVPR 2020)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size
        kernel_size = int(abs((torch.log2(torch.tensor(channels).float()) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Global average pooling
        y = self.avg_pool(x).view(b, 1, c)

        # 1D convolution
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


def get_attention_module(
    attention_type: str,
    channels: int,
    reduction: int = 16
) -> nn.Module:
    """
    Factory function to get attention module by type.

    Args:
        attention_type: Type of attention ('cbam', 'se', 'eca', 'none')
        channels: Number of input channels
        reduction: Reduction ratio for channel attention

    Returns:
        Attention module or Identity if 'none'
    """
    attention_modules = {
        'cbam': lambda: CBAM(channels, reduction),
        'se': lambda: SEBlock(channels, reduction),
        'eca': lambda: ECABlock(channels),
        'none': lambda: nn.Identity()
    }

    if attention_type not in attention_modules:
        raise ValueError(f"Unknown attention type: {attention_type}")

    return attention_modules[attention_type]()

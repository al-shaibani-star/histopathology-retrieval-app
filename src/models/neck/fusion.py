"""
Feature Fusion Neck
رقبة دمج الميزات

Implements Adaptive Weighted Fusion for multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AdaptiveWeightedFusionNeck(nn.Module):
    """
    Adaptive Weighted Feature Fusion Neck

    Fuses multi-scale features from backbone using learnable weights.

    Features:
        - Lateral connections to unify channel dimensions
        - Learnable fusion weights (softmax normalized)
        - Supports different fusion strategies
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        fusion_type: str = 'adaptive'
    ):
        """
        Args:
            in_channels_list: List of input channel dimensions from each stage
            out_channels: Output channel dimension
            fusion_type: Fusion strategy ('adaptive', 'concat', 'add')
        """
        super().__init__()

        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.fusion_type = fusion_type

        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for in_ch in in_channels_list
        ])

        # Fusion mechanism
        if fusion_type == 'adaptive':
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_levels) / self.num_levels
            )
        elif fusion_type == 'concat':
            # Reduction after concatenation
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(out_channels * self.num_levels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )

        # Output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from backbone stages

        Returns:
            Fused feature tensor (B, out_channels, H, W)
        """
        assert len(features) == self.num_levels

        # Apply lateral convolutions
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Get target size (use smallest feature map)
        target_size = laterals[-1].shape[2:]

        # Resize all features to target size
        resized = [
            F.interpolate(lat, size=target_size, mode='bilinear', align_corners=False)
            for lat in laterals
        ]

        # Fusion
        if self.fusion_type == 'adaptive':
            # Weighted sum with learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, resized))

        elif self.fusion_type == 'concat':
            # Concatenate and reduce
            fused = torch.cat(resized, dim=1)
            fused = self.reduce_conv(fused)

        elif self.fusion_type == 'add':
            # Simple addition
            fused = sum(resized) / len(resized)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Output refinement
        output = self.output_conv(fused)

        return output

    def get_fusion_weights(self) -> torch.Tensor:
        """Get normalized fusion weights (for visualization)."""
        if self.fusion_type == 'adaptive':
            return F.softmax(self.fusion_weights, dim=0)
        return None


class FPN(nn.Module):
    """
    Feature Pyramid Network

    Top-down pathway with lateral connections for multi-scale features.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        super().__init__()

        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            features: List of feature tensors from backbone

        Returns:
            List of FPN feature tensors
        """
        # Lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway
        for i in range(self.num_levels - 2, -1, -1):
            size = laterals[i].shape[2:]
            upsampled = F.interpolate(laterals[i + 1], size=size, mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + upsampled

        # Output convolutions
        outputs = [
            conv(lat) for conv, lat in zip(self.output_convs, laterals)
        ]

        return outputs

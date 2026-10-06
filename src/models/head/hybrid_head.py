"""
Hybrid Representation Head
رأس التمثيل الهجين

Produces 18,496-dimensional feature vector:
- Global: 2,048-dim (semantic features)
- Spatial: 16,384-dim (spatial layout)
- Attention: 64-dim (important regions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class GlobalBranch(nn.Module):
    """
    Global Feature Branch

    Extracts global semantic features using adaptive pooling.
    Output: 2,048-dimensional vector
    """

    def __init__(self, in_channels: int, out_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc(x)
        return x


class SpatialBranch(nn.Module):
    """
    Spatial Feature Branch

    Preserves spatial information through grid pooling.
    Output: 16,384-dimensional vector
    """

    def __init__(self, in_channels: int, out_dim: int = 16384, dropout: float = 0.1):
        super().__init__()

        # Calculate grid size
        self.grid_size = int(np.sqrt(out_dim // in_channels))
        actual_dim = in_channels * self.grid_size * self.grid_size

        self.pool = nn.AdaptiveAvgPool2d(self.grid_size)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(actual_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc(x)
        return x


class AttentionBranch(nn.Module):
    """
    Attention Feature Branch

    Captures attention-weighted features.
    Output: 64-dimensional vector
    """

    def __init__(self, in_channels: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.feature_conv = nn.Conv2d(in_channels, out_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention weights
        attention_weights = self.attention(x)

        # Apply attention
        attended = x * attention_weights

        # Extract features
        features = self.feature_conv(attended)
        features = self.pool(features)
        features = self.fc(features)

        return features, attention_weights


class HybridRepresentationHead(nn.Module):
    """
    Hybrid Representation Head

    Combines global, spatial, and attention features into a
    comprehensive 18,496-dimensional representation.

    Architecture:
        Input (B, C, H, W)
            ├── Global Branch    → 2,048-dim
            ├── Spatial Branch   → 16,384-dim
            └── Attention Branch → 64-dim
                                   ─────────
                        Combined → 18,496-dim
    """

    def __init__(
        self,
        in_channels: int = 256,
        global_dim: int = 2048,
        spatial_dim: int = 16384,
        attention_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # Feature branches
        self.global_branch = GlobalBranch(in_channels, global_dim, dropout)
        self.spatial_branch = SpatialBranch(in_channels, spatial_dim, dropout)
        self.attention_branch = AttentionBranch(in_channels, attention_dim, dropout)

        # Dimensions
        self.global_dim = global_dim
        self.spatial_dim = spatial_dim
        self.attention_dim = attention_dim
        self.total_dim = global_dim + spatial_dim + attention_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary containing:
                - 'global': Global features (B, 2048)
                - 'spatial': Spatial features (B, 16384)
                - 'attention': Attention features (B, 64)
                - 'attention_map': Attention weights (B, 1, H, W)
                - 'combined': Concatenated features (B, 18496)
        """
        # Extract features from each branch
        global_feat = self.global_branch(x)
        spatial_feat = self.spatial_branch(x)
        attention_feat, attention_map = self.attention_branch(x)

        # Concatenate all features
        combined = torch.cat([global_feat, spatial_feat, attention_feat], dim=1)

        return {
            'global': global_feat,
            'spatial': spatial_feat,
            'attention': attention_feat,
            'attention_map': attention_map,
            'combined': combined
        }

    def get_feature_dims(self) -> Dict[str, int]:
        """Get dimensions of each feature type."""
        return {
            'global': self.global_dim,
            'spatial': self.spatial_dim,
            'attention': self.attention_dim,
            'total': self.total_dim
        }

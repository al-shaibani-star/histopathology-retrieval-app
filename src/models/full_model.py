"""
Full Medical Image Analysis Model
النموذج الكامل لتحليل الصور الطبية
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .backbone import ConvNeXtV2Backbone
from .neck import AdaptiveWeightedFusionNeck
from .head import HybridRepresentationHead


class MedicalImageModel(nn.Module):
    """
    Medical Image Analysis Model

    Complete model architecture for medical image analysis:
        Input → Backbone → Neck → Head → Classifier

    Features:
        - ConvNeXt V2 backbone with attention
        - Adaptive weighted feature fusion
        - Hybrid representation (18,496-dim)
        - Classification head
    """

    def __init__(
        self,
        num_classes: int,
        # Backbone params
        in_channels: int = 3,
        base_channels: int = 64,
        depths: List[int] = None,
        use_attention: bool = True,
        attention_type: str = 'cbam',
        # Neck params
        neck_channels: int = 256,
        fusion_type: str = 'adaptive',
        # Head params
        global_dim: int = 2048,
        spatial_dim: int = 16384,
        attention_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]

        self.num_classes = num_classes

        # Backbone
        self.backbone = ConvNeXtV2Backbone(
            in_channels=in_channels,
            base_channels=base_channels,
            depths=depths,
            use_attention=use_attention,
            attention_type=attention_type
        )

        # Neck
        self.neck = AdaptiveWeightedFusionNeck(
            in_channels_list=self.backbone.out_channels,
            out_channels=neck_channels,
            fusion_type=fusion_type
        )

        # Head
        self.head = HybridRepresentationHead(
            in_channels=neck_channels,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.head.total_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Feature dimension
        self.feature_dim = self.head.total_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - 'logits': Classification logits (B, num_classes)
                - 'features': Combined features (B, 18496)
                - 'global_features': Global features (B, 2048)
                - 'attention_map': Attention weights (B, 1, H, W)
        """
        # Backbone: Extract multi-scale features
        backbone_features = self.backbone(x)

        # Neck: Fuse features
        neck_features = self.neck(backbone_features)

        # Head: Generate hybrid representation
        head_output = self.head(neck_features)

        # Classifier
        logits = self.classifier(head_output['combined'])

        output = {
            'logits': logits,
            'features': head_output['combined'],
            'global_features': head_output['global'],
            'attention_map': head_output['attention_map']
        }

        if return_features:
            output['backbone_features'] = backbone_features
            output['neck_features'] = neck_features
            output['spatial_features'] = head_output['spatial']
            output['attention_features'] = head_output['attention']

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features only (for similarity search).

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature tensor (B, 18496)
        """
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        head_output = self.head(neck_features)
        return head_output['combined']

    def get_embeddings(
        self,
        x: torch.Tensor,
        level: str = 'combined'
    ) -> torch.Tensor:
        """
        Get embeddings at different levels.

        Args:
            x: Input tensor
            level: 'combined', 'global', 'spatial', or 'attention'

        Returns:
            Feature tensor at specified level
        """
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        head_output = self.head(neck_features)

        if level == 'combined':
            return head_output['combined']
        elif level == 'global':
            return head_output['global']
        elif level == 'spatial':
            return head_output['spatial']
        elif level == 'attention':
            return head_output['attention']
        else:
            raise ValueError(f"Unknown level: {level}")


def create_model(
    num_classes: int,
    model_size: str = 'base',
    pretrained: bool = False
) -> MedicalImageModel:
    """
    Factory function to create model.

    Args:
        num_classes: Number of output classes
        model_size: 'tiny', 'small', 'base', or 'large'
        pretrained: Whether to load pretrained weights

    Returns:
        MedicalImageModel instance
    """
    configs = {
        # For GPUs with < 4GB memory
        'micro': {
            'base_channels': 16,
            'depths': [1, 1, 1, 1],
            'neck_channels': 64,
            'global_dim': 256,
            'spatial_dim': 512,
            'attention_dim': 16,
            'use_attention': False  # Disable attention for memory savings
        },
        'tiny': {
            'base_channels': 32,
            'depths': [1, 1, 2, 1],
            'neck_channels': 128,
            'global_dim': 512,
            'spatial_dim': 2048,
            'attention_dim': 32
        },
        'small': {
            'base_channels': 48,
            'depths': [2, 2, 4, 2],
            'neck_channels': 192,
            'global_dim': 1024,
            'spatial_dim': 8192,
            'attention_dim': 48
        },
        'base': {
            'base_channels': 64,
            'depths': [2, 2, 6, 2],
            'neck_channels': 256,
            'global_dim': 2048,
            'spatial_dim': 16384,
            'attention_dim': 64
        },
        'large': {
            'base_channels': 96,
            'depths': [3, 3, 9, 3],
            'neck_channels': 384,
            'global_dim': 4096,
            'spatial_dim': 32768,
            'attention_dim': 128
        }
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]

    model = MedicalImageModel(
        num_classes=num_classes,
        **config
    )

    if pretrained:
        # Load pretrained weights (to be implemented)
        print(f"Loading pretrained weights for {model_size} model...")

    return model

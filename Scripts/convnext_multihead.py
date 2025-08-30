import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

# torchvision 버전별 가중치 로딩 호환
try:
    from torchvision.models import ConvNeXt_Tiny_Weights
    _WEIGHTS = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    _KW = dict(weights=_WEIGHTS)
except Exception:
    _KW = dict(pretrained=True)

class ConvNeXtTinyMultiHead(nn.Module):
    """ConvNeXt-Tiny 백본 + (nudity, violence) 2-헤드 이진 로짓"""
    def __init__(self):
        super().__init__()
        base = convnext_tiny(**_KW)
        # 풀링 전까지를 백본으로 추출
        self.backbone = nn.Sequential(
            base.features,         # [B, C, H, W]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),          # [B, C]
        )
        # 최종 차원(보통 768)
        try:
            dim = base.classifier[2].in_features
        except Exception:
            # 혹시 구조가 다른 경우 대비
            dim = list(base.classifier.modules())[-1].in_features

        self.head_nudity   = nn.Linear(dim, 1)
        self.head_violence = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        nud = self.head_nudity(z)     # [B,1]
        vio = self.head_violence(z)   # [B,1]
        return torch.cat([nud, vio], dim=1)  # [B,2]

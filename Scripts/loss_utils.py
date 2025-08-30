import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    BCEWithLogitsLoss 버전인데, 라벨 -1은 무시합니다.
    멀티헤드 쓰려고 걍 -1 해둠
    
    Args:
        logits: [B, 2]  → 모델 출력 (nudity, violence)
        targets: [B, 2] → 라벨 {0, 1, -1}, -1은 무시
    
    Returns:
        loss (scalar tensor)
    """
    # 마스크: -1이 아닌 위치만 학습에 반영
    mask = (targets >= 0).float()              # [B,2]
    
    # -1 → 0으로 클램프 (무시할 거라 값은 중요하지 않음)
    targets_pos = torch.clamp(targets, min=0).float()
    
    # BCE loss per element
    loss = F.binary_cross_entropy_with_logits(
        logits, targets_pos, reduction="none"
    )  # [B,2]
    
    # 마스크 적용 후 평균
    masked_loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return masked_loss

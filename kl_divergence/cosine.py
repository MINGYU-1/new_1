import math

def gamma_cosine(epoch: int,
                 gamma_max: float,
                 warmup_epochs: int,
                 fix_after: bool = True) -> float:
    """
    epoch: 1부터 시작
    gamma_max: 최종 KL 가중치
    warmup_epochs: gamma를 0→gamma_max로 증가시키는 구간 길이
    fix_after: warmup 이후 gamma_max로 고정 여부
    """
    if epoch <= warmup_epochs:
        return gamma_max * 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
    
    if fix_after:
        return gamma_max
    
    return gamma_max

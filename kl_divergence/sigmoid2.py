import math

def gamma_sigmoid_fixed(epoch: int,
                        gamma_max: float,
                        center: int,
                        steepness: float = 0.3,
                        fix_epoch: int = None) -> float:
    """
    epoch: 1부터 시작
    gamma_max: 최종 KL 가중치
    center: sigmoid 중간 지점
    steepness: 증가 속도
    fix_epoch: 이 epoch 이후 gamma_max로 고정
    """
    gamma = gamma_max / (1.0 + math.exp(-steepness * (epoch - center)))

    if fix_epoch is not None and epoch >= fix_epoch:
        return gamma_max

    return gamma

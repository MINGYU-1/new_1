    """
    epoch: 1부터 시작
    gamma_max: 최종 KL 가중치
    warmup_epochs: 몇 epoch 동안 0 -> gamma_max로 선형 증가할지
    """
def gamma_linear(epoch: int, gamma_max: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return gamma_max
    return gamma_max * min(epoch / warmup_epochs, 1.0)

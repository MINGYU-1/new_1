"""
    epoch: 1부터 시작
    gamma_max: 각 사이클에서 도달할 KL 최대 가중치
    cycle_len: 한 사이클 길이 (epoch)
    ratio: 사이클 중 몇 % 구간에서 0->gamma_max로 올릴지
           나머지 구간은 gamma_max 유지
    """
def gamma_cyclical(epoch: int, gamma_max: float, cycle_len: int, ratio: float = 0.5) -> float:
    if cycle_len <= 0:
        return gamma_max
    pos = (epoch - 1) % cycle_len + 1  # 1..cycle_len
    ramp = max(int(cycle_len * ratio), 1)
    if pos <= ramp:
        return gamma_max * (pos / ramp)
    return gamma_max

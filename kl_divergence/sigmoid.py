import math
    """
    epoch: 1부터 시작
    gamma_max: 최종 KL 가중치
    center: sigmoid가 중간(0.5*gamma_max)쯤 되는 epoch
    steepness: 커질수록 증가가 급격해짐
    """ 
def gamma_sigmoid(epoch: int, gamma_max: float, center: int, steepness: float = 0.3) -> float:
    return gamma_max / (1.0 + math.exp(-steepness * (epoch - center)))

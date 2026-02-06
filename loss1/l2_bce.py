import torch
import torch.nn.functional as F
def l2_bce(binary_logit,x, mu, logvar,beta=1.0, gamma=1.0):


    # 1. Classification Loss (BCE): 금속 존재 여부 (이미지의 probability 부분)
    x_binary = (x > 0).float()
    bce_loss = F.binary_cross_entropy_with_logits(
    binary_logit,
    x_binary,
    reduction='sum',
    )


   

    # 3. KL Divergence: Latent Space 정규화
    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    batch_size = x.shape[0]
    # 최종 손실 합산 (가중치 조절)
    # 각 loss를 batch_size로 나누어 평균 손실을 구함
    total_loss = (beta * bce_loss + gamma * kl_loss)/batch_size

    return {
        'loss': total_loss,
        'bce_loss': bce_loss/batch_size,
        'kl_loss': kl_loss/batch_size,
    }
import torch
import torch.nn.functional as F
def loss1(binary_logit, x_hat, x,x2, mu, logvar, alpha=1.0,beta=1.0, gamma=1.0):
    
    batch_size = x.shape[0]

    # 1. Classification Loss (BCE): 금속 존재 여부 (이미지의 probability 부분)
    x_binary = (x > 0).float()
    bce_loss = F.binary_cross_entropy_with_logits(binary_logit, x_binary, reduction='sum')

    # 2. Reconstruction Loss (MSE): 금속의 수치 정보
    # 실제 존재하는 부분에 대해서만 MSE를 계산하는 것이 더 정확할 수 있습니다.
    mse_loss = F.mse_loss(x_hat, x2, reduction='sum')

    # 3. KL Divergence: Latent Space 정규화
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    # 최종 손실 합산 (가중치 조절)
    # 각 loss를 batch_size로 나누어 평균 손실을 구함
    total_loss = (alpha* mse_loss + beta * bce_loss + gamma * kl_loss) / batch_size

    return {
        'loss': total_loss,
        'bce_loss': bce_loss / batch_size,
        'mse_loss': mse_loss / batch_size,
        'kl_loss': kl_loss / batch_size,
    }
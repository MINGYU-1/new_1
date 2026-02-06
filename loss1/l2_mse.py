import torch
import torch.nn.functional as F
def l2_mse(x_hat, x, mu, logvar, 
                       alpha=1.0,gamma=1.0):


    mse_loss = F.mse_loss(x_hat, x, reduction='sum')

    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    batch_size = x.shape[0]
    total_loss = (alpha * mse_loss + gamma * kl_loss)/batch_size
    
    return {
        'loss': total_loss,
        'mse_loss': mse_loss/batch_size ,
        'kl_loss': kl_loss/batch_size
    }
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEcVAE(nn.Module):
    def __init__(self,x_dim,c_dim,z_dim, h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
    
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)


        ## decoder_mse[z+c]->recon(x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )
    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps
    
    def forward(self,x,c):
        h = self.encoder(torch.cat([x, c], dim=1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)


        x_hat = self.decoder_mse(torch.concat([z,c],dim = 1))

        return x_hat, z_mu,z_logvar
    


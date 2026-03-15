import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSelfRepresentation(nn.Module):

    def __init__(self,n_samples,latent_dim,lambda1=1e-3,lambda2=1e-3):

        super().__init__()

        self.Z = nn.Parameter(torch.zeros(n_samples,n_samples))

        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self,H_latent):

        n = H_latent.shape[0]

        Z_non_diag = self.Z * (1 - torch.eye(n,device=H_latent.device))

        H_rec = Z_non_diag @ H_latent

        recon_loss = F.mse_loss(H_rec,H_latent)

        l1_loss = self.lambda1 * torch.sum(torch.abs(Z_non_diag))

        nuclear_loss = self.lambda2 * torch.norm(Z_non_diag,p='nuc')

        loss = recon_loss + l1_loss + nuclear_loss

        return H_rec,loss
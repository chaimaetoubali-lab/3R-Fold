import torch
from torch import nn

class RNADeepAE(nn.Module):

    def __init__(self, input_dim=128, latent_dim=64):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,input_dim)
        )

    def forward(self,x):

        z = self.encoder(x)

        x_rec = self.decoder(z)

        return x_rec,z
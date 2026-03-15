import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.1):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature

    labels = torch.arange(z1.size(0)).to(z1.device)

    loss = F.cross_entropy(logits, labels)

    return loss
import torch

def topk_from_sr(Z,k=5):

    n = Z.shape[0]

    Z_final = Z * (1 - torch.eye(n))

    z_query = Z_final[0,1:]

    vals,idx = torch.topk(z_query,k)

    return vals,idx
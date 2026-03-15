import torch
from models.embedder import seq_to_tensor

def prepare_query(seq, device):

    tokens = seq_to_tensor(seq, device)

    return tokens.unsqueeze(0)
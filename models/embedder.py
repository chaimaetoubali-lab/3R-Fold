import torch
import torch.nn as nn

nt_map = {"A":1, "C":2, "G":3, "U":4, "T":4, "N":0}

def seq_to_tensor(seq, device):
    return torch.tensor([nt_map.get(nt,0) for nt in seq], dtype=torch.long, device=device)

class RNAEmbedder(nn.Module):

    def __init__(self, d_model=128, nhead=4, num_layers=2, vocab_size=5):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, seq_tokens):

        x = self.embedding(seq_tokens)

        x = self.transformer(x)

        e = x.mean(dim=1)

        return e
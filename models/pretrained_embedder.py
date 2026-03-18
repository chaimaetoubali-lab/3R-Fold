import torch
from fm import pretrained
import os
os.getcwd()


class PretrainedRNAEmbedder:
    def __init__(self, device="cpu", max_length=1022):
        self.device = device
        self.max_length = max_length

        self.model, self.alphabet = pretrained.rna_fm_t12()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

    def _clean_sequence(self, seq: str) -> str:
        seq = seq.upper().replace("T", "U")
        allowed = {"A", "C", "G", "U", "N"}
        seq = "".join(ch if ch in allowed else "N" for ch in seq)
        return seq[: self.max_length]

    def embed(self, sequences):
        cleaned = [self._clean_sequence(seq) for seq in sequences]
        batch = [(str(i), seq) for i, seq in enumerate(cleaned)]

        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            out = self.model(tokens, repr_layers=[12])

        reps = out["representations"][12]

        lengths = (tokens != self.alphabet.padding_idx).sum(dim=1)

        embeddings = []
        for i, length in enumerate(lengths):
            start = 1
            end = max(start + 1, int(length.item()) - 1)
            embeddings.append(reps[i, start:end].mean(dim=0))

        return torch.stack(embeddings, dim=0)
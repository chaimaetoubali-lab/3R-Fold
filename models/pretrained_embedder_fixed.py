import torch
import os
import requests
import hashlib
from fm import pretrained


class PretrainedRNAEmbedder:
    def __init__(self, device="cpu", max_length=1022):
        self.device = device
        self.max_length = max_length

        # Handle corrupted model download
        self._ensure_model_downloaded()
        self.model, self.alphabet = pretrained.rna_fm_t12()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

    def _ensure_model_downloaded(self):
        """Ensure the RNA-FM model is properly downloaded and not corrupted"""
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        model_path = os.path.join(cache_dir, "RNA-FM_pretrained.pth")
        
        # Check if file exists and is valid
        if os.path.exists(model_path):
            try:
                # Try to load the file to verify it's not corrupted
                torch.load(model_path, map_location='cpu')
                print(f"Model file verified: {model_path}")
                return
            except Exception as e:
                print(f"Corrupted model file detected: {e}")
                print("Removing corrupted file and re-downloading...")
                os.remove(model_path)
        
        # Download the model manually with better error handling
        self._download_model(model_path)

    def _download_model(self, model_path):
        """Download RNA-FM model with retry logic"""
        url = "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"Downloading RNA-FM model from {url}")
        print(f"Saving to: {model_path}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
            
            print("\nDownload completed!")
            
            # Verify the downloaded file
            try:
                torch.load(model_path, map_location='cpu')
                print("Model file verification successful!")
            except Exception as e:
                print(f"Downloaded file is corrupted: {e}")
                os.remove(model_path)
                raise RuntimeError("Failed to download valid model file")
                
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise

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

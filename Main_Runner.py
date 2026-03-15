# %%
import sys, importlib

from config import get_results_path

sys.path.insert(0, "/content/drive/MyDrive/Foldv")

import data.build_dataset as bd
importlib.reload(bd)

dataset = bd.build_dataset()
print("Final size:", len(dataset))
print("Families:", sorted(set(dataset["family"])))

# %%
import data.build_dataset as bd
print("Imported from:", bd.__file__)

dataset = bd.build_dataset()
print("Final size:", len(dataset))
print("Families:", sorted(set(dataset["family"])))
# %%
import os

from data.build_dataset import build_dataset
from models.pretrained_embedder import PretrainedRNAEmbedder
from retrieval.build_kb import build_kb_embeddings
from config import  get_results_path

os.makedirs(get_results_path(), exist_ok=True)

dataset = build_dataset()
print("Dataset size:", len(dataset))
print("Families:", sorted(set(dataset["family"])))

embedder = PretrainedRNAEmbedder(device="cpu", max_length=256)

emb_path = os.path.join(get_results_path(), "kb_embeddings_rnafm_len256_full.pt")

embeddings = build_kb_embeddings(
    dataset,
    embedder,
    batch_size=4,
    max_length=256,
    cache_path=emb_path,
)

print("Saved embeddings to:", emb_path)
print("Embeddings shape:", embeddings.shape)
# %%
from experiments.run_dsr_global_all import main as gsrg_main
from experiments.run_dsr_faiss_hybrid import main as hybrid_main
from experiments.run_ablation import main as ablation_main
from experiments.run_benchmark import main as benchmark_main
from experiments.run_lofo_benchmark import run_lofo

#gsrg_main()
#hybrid_main()
ablation_main()
benchmark_main()
run_lofo()
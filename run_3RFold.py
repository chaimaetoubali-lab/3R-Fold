import yaml
import torch

from data.build_dataset import build_dataset
from retrieval.build_kb import build_kb_embeddings
from retrieval.faiss_index import FaissIndex
from models.pretrained_embedder import PretrainedRNAEmbedder
from retrieval.retrieval_guided_fold import fold_with_neighbor_guidance
from evaluation.metrics import evaluate_structure

def main(config_path="config/default.yaml"):
    config = yaml.safe_load(open(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Building dataset...")
    dataset = build_dataset()

    # Debug mode: décommente si besoin
    # dataset = dataset.select(range(200))

    print("Dataset size:", len(dataset))

    print("Initializing pretrained embedder...")
    
    embedding_cfg = config.get("embedding", {})

    embedder = PretrainedRNAEmbedder(
    device=device,
    max_length=embedding_cfg.get("max_length", 256))


    print("Building / loading embeddings...")
    kb_embeddings = build_kb_embeddings(
        dataset,
        embedder,
        batch_size=embedding_cfg.get("batch_size", 4),
        max_length=embedding_cfg.get("max_length", 256),
        cache_path=embedding_cfg.get("cache_path", "kb_embeddings_rnafm_len256.pt"), )

    kb_embeddings = kb_embeddings.to(device)
    print("KB embeddings shape:", kb_embeddings.shape)

    print("Building FAISS index...")
    index = FaissIndex(kb_embeddings.shape[1])
    index.build(kb_embeddings)

    query = dataset[0]
    print("Query family:", query["family"])

    query_embedding = embedder.embed([query["sequence"]]).to(device)
    print("Query embedding shape:", query_embedding.shape)

    top_k = config["retrieval"].get("top_k", 5)

    distances, indices = index.search(query_embedding, k=top_k)

    neighbor_indices = [int(i) for i in indices[0]]
    neighbors = [dataset[i] for i in neighbor_indices]

    print("Neighbor families:", [n["family"] for n in neighbors])

    # Convertir distances FAISS -> similarités
    # Ici index L2, donc on transforme grossièrement en poids inverses
    raw_distances = distances[0]
    similarity_weights = [1.0 / (float(d) + 1e-8) for d in raw_distances]

    neighbor_structures = []
    valid_weights = []

    for neighbor, weight in zip(neighbors, similarity_weights):
        struct = neighbor.get("structure", "")
        if struct and isinstance(struct, str):
            neighbor_structures.append(struct)
            valid_weights.append(weight)

    print("Number of neighbor structures used:", len(neighbor_structures))

    pred_structure = fold_with_neighbor_guidance(
     query["sequence"],
     neighbor_structures,
     weights=valid_weights,
     min_pair_freq=config.get("folding", {}).get("min_pair_freq", 0.4),
     unpaired_bonus=config.get("folding", {}).get("unpaired_bonus", 0.3),)


    sens, ppv, f1 = evaluate_structure(pred_structure, query["structure"])

    results = {
        "query_family": query["family"],
        "neighbor_families": [n["family"] for n in neighbors],
        "pred_structure": pred_structure,
        "ref_structure": query["structure"],
        "sensitivity": sens,
        "ppv": ppv,
        "f1": f1,
    }

    print(results)


if __name__ == "__main__":
    main("config/default.yaml")
#python run_3RFold.py --config config/default.yaml
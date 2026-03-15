import random
import pandas as pd

from evaluation.metrics import evaluate_structure
from evaluation.retrieval_metrics import family_purity, avg_similarity_from_distances, top1_family_match
from evaluation.runtime_analysis import Timer
from retrieval.retrieval_guided_fold import fold_with_neighbor_guidance
from retrieval.faiss_index import FaissIndex
from folding.template_guided_fold import fold_with_template_transfer

def vienna_only_fold(sequence):
    import RNA
    fc = RNA.fold_compound(sequence)
    structure, _ = fc.mfe()
    return structure


def split_dataset(dataset, test_fraction=0.2, seed=42):
    data = list(dataset)
    random.Random(seed).shuffle(data)
    split_idx = int((1.0 - test_fraction) * len(data))
    kb_data = data[:split_idx]
    test_data = data[split_idx:]
    return kb_data, test_data


def build_faiss_for_kb(kb_data, embedder, batch_size=4, max_length=256, cache_path=None):
    from retrieval.build_kb import build_kb_embeddings

    kb_embeddings = build_kb_embeddings(
        kb_data,
        embedder,
        batch_size=batch_size,
        max_length=max_length,
        cache_path=cache_path,
    )

    index = FaissIndex(kb_embeddings.shape[1])
    index.build(kb_embeddings)
    return kb_embeddings, index


def get_neighbors_faiss(query_sequence, query_embedding, kb_data, index, k):
    distances, indices = index.search(query_embedding, k=k)

    neighbor_indices = [int(i) for i in indices[0]]
    neighbors = [kb_data[i] for i in neighbor_indices]
    neighbor_families = [n["family"] for n in neighbors]

    raw_distances = distances[0]
    weights = [1.0 / (float(d) + 1e-8) for d in raw_distances]

    return neighbors, neighbor_families, raw_distances, weights


def get_neighbors_oracle(query_item, kb_data, k):
    same_family = [x for x in kb_data if x["family"] == query_item["family"]]
    neighbors = same_family[:k]
    neighbor_families = [n["family"] for n in neighbors]
    raw_distances = [0.0] * len(neighbors)
    weights = [1.0] * len(neighbors)
    return neighbors, neighbor_families, raw_distances, weights


def run_single_query(
    query_item,
    kb_data,
    embedder,
    method_name,
    retrieval_type="faiss",
    index=None,
    k=5,
    use_guidance=True,
    guidance_mode="consensus",
):
    timer = Timer()

    query_id = query_item["id"]
    query_family = query_item["family"]
    query_seq = query_item["sequence"]
    ref_structure = query_item["structure"]

    timer.start("embed_query")
    query_embedding = embedder.embed([query_seq])
    timer.stop("embed_query")

    neighbors = []
    neighbor_families = []
    raw_distances = []
    weights = []

    if retrieval_type == "faiss":
        timer.start("retrieval")
        neighbors, neighbor_families, raw_distances, weights = get_neighbors_faiss(
            query_seq,
            query_embedding,
            kb_data,
            index,
            k,
        )
        timer.stop("retrieval")

    elif retrieval_type == "oracle":
        timer.start("retrieval")
        neighbors, neighbor_families, raw_distances, weights = get_neighbors_oracle(
            query_item,
            kb_data,
            k,
        )
        timer.stop("retrieval")

    else:
        timer.start("retrieval")
        timer.stop("retrieval")

        timer.start("folding")
    if use_guidance and neighbors:
        neighbor_structures = []
        valid_weights = []
        valid_neighbors = []

        for n, w in zip(neighbors, weights):
            struct = n.get("structure", "")
            if struct and isinstance(struct, str):
                neighbor_structures.append(struct)
                valid_weights.append(w)
                valid_neighbors.append(n)

        if guidance_mode == "template_transfer":
            pred_structure = fold_with_template_transfer(
                query_seq,
                valid_neighbors,
                weights=valid_weights,
                min_pair_freq=0.3,
            )
        else:
            pred_structure = fold_with_neighbor_guidance(
                query_seq,
                neighbor_structures,
                weights=valid_weights,
                min_pair_freq=0.4,
                unpaired_bonus=0.3,
            )
    else:
        pred_structure = vienna_only_fold(query_seq)
    timer.stop("folding")

    sens, ppv, f1 = evaluate_structure(pred_structure, ref_structure)

    row = {
        "query_id": query_id,
        "family": query_family,
        "length": len(query_seq),
        "method": method_name,
        "retrieval_type": retrieval_type,
        "use_guidance": int(use_guidance),
        "guidance_mode": guidance_mode,
        "use_dsr": 0,
        "k": k,
        "neighbor_families": "|".join(neighbor_families),
        "family_purity": family_purity(query_family, neighbor_families),
        "top1_family_match": top1_family_match(query_family, neighbor_families),
        "avg_similarity": avg_similarity_from_distances(raw_distances, metric="l2"),
        "sensitivity": sens,
        "ppv": ppv,
        "f1": f1,
        "sequence": query_seq,
        "ref_structure": ref_structure,
        "pred_structure": pred_structure,
        "neighbor_structures": "|||".join(neighbor_structures) if use_guidance and neighbors else "",
        "embed_query_time": timer.get("embed_query"),
        "retrieval_time": timer.get("retrieval"),
        "folding_time": timer.get("folding"),
        "total_time": timer.get("embed_query") + timer.get("retrieval") + timer.get("folding"),
    }
    return row


def run_benchmark(
    dataset,
    embedder,
    n_queries=50,
    test_fraction=0.2,
    seed=42,
    method_name="rnafm_faiss_guided",
    retrieval_type="faiss",
    use_guidance=True,
    guidance_mode="consensus",
    k=5,
    batch_size=4,
    max_length=256,
    cache_path=None,
):
    kb_data, test_data = split_dataset(dataset, test_fraction=test_fraction, seed=seed)

    if n_queries is not None:
        test_data = test_data[:n_queries]

    index = None
    if retrieval_type == "faiss":
        _, index = build_faiss_for_kb(
            kb_data,
            embedder,
            batch_size=batch_size,
            max_length=max_length,
            cache_path=cache_path,
        )

    rows = []
    for query_item in test_data:
        row = run_single_query(
            query_item=query_item,
            kb_data=kb_data,
            embedder=embedder,
            method_name=method_name,
            retrieval_type=retrieval_type,
            index=index,
            k=k,
            use_guidance=use_guidance,
            guidance_mode=guidance_mode,
        )
        rows.append(row)

    return pd.DataFrame(rows)
import numpy as np


def family_purity(query_family, neighbor_families):
    if not neighbor_families:
        return 0.0
    return sum(1 for f in neighbor_families if f == query_family) / len(neighbor_families)


def avg_similarity_from_distances(distances, metric="l2"):
    if distances is None or len(distances) == 0:
        return 0.0

    distances = np.asarray(distances, dtype=float)

    if metric == "l2":
        sims = 1.0 / (distances + 1e-8)
        return float(np.mean(sims))

    return float(np.mean(distances))


def top1_family_match(query_family, neighbor_families):
    if not neighbor_families:
        return 0
    return int(neighbor_families[0] == query_family)
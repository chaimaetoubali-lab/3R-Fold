import torch


def retrieve_neighbors_dsr(Z, kb_data, k=5):
    """
    Use Deep Self Representation matrix to select neighbors.
    Assumes query is index 0 in Z.
    """

    n = Z.shape[0]

    Z_final = Z * (1 - torch.eye(n))

    z_query = Z_final[0, 1:]

    vals, idx = torch.topk(z_query, k)

    neighbor_indices = idx.tolist()

    neighbors = [kb_data[i] for i in neighbor_indices]
    neighbor_families = [n["family"] for n in neighbors]

    return neighbors, neighbor_families, vals
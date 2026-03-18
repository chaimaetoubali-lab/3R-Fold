import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_rna_retrieval_graph(Z, families, n=200, k=5):
    Z = np.asarray(Z)
    n = min(n, len(families), Z.shape[0])

    G = nx.Graph()

    # add nodes only by index
    for i in range(n):
        G.add_node(i)

    # add top-k neighbor edges
    for i in range(n):
        neighbors = np.argsort(Z[i])[-(k + 1):-1]
        for j in neighbors:
            j = int(j)
            if i != j and j < n:
                G.add_edge(i, j)

    node_list = list(G.nodes())
    node_families = [families[i] for i in node_list]

    unique_families = sorted(set(node_families))
    family_to_color = {fam: idx for idx, fam in enumerate(unique_families)}
    node_colors = [family_to_color[f] for f in node_families]

    config = yaml.safe_load(open("config/default.yaml"))
    seed = config.get("seed", 42)

    pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        node_size=40,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.15,
        width=0.8,
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=fam,
            markerfacecolor=plt.cm.tab10(
                family_to_color[fam] / max(1, len(unique_families) - 1)
            ),
            markersize=8,
        )
        for fam in unique_families
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.title("RNA Retrieval Neighborhood Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
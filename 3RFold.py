# %%
from run_3RFold import main

main(config_path="config/default.yaml")
# %% [markdown]
# # Run benchmark and save results
# %%
from experiments.run_benchmark import main
main()
# %% [markdown]
# # Run ablation and save results
# %%
from experiments.run_ablation import main
main()
# %% [markdown]
# # Plotting results
# %%
from visualisation.performance_plots import *
from visualisation.family_plots import *
from visualisation.retrieval_plots import *
from visualisation.improvement_plots import *


plot_f1_by_method("results/ablation_results.csv")
plot_k_sweep("results/ablation_results.csv")
plot_violin_f1("results/ablation_results.csv")
plot_family_f1("results/ablation_results.csv")
plot_purity_vs_f1("results/ablation_results.csv")
plot_similarity_vs_f1("results/ablation_results.csv")
plot_improvement_over_baseline("results/ablation_results.csv")
plot_improvement_vs_purity("results/ablation_results.csv")

plot_f1_by_method("results/ablation_results.csv")
plot_k_sweep("results/ablation_results.csv")
plot_violin_f1("results/ablation_results.csv")
plot_family_f1("results/ablation_results.csv")
plot_purity_vs_f1("results/ablation_results.csv")
plot_similarity_vs_f1("results/ablation_results.csv")

plot_f1_by_method("results/ablation_results.csv")
plot_k_sweep("results/ablation_results.csv")
plot_violin_f1("results/ablation_results.csv")

plot_family_f1("results/ablation_results.csv")

plot_purity_vs_f1("results/ablation_results.csv")
plot_similarity_vs_f1("results/ablation_results.csv")
# %%
import importlib
import visualisation.length_plots as lp

importlib.reload(lp)

lp.plot_improvement_vs_length("results/ablation_results.csv")
# %% [markdown]
# # Run LOFO benchmark
# %%
from experiments.run_lofo_benchmark import run_lofo
run_lofo()
# %%
from evaluation.generalization_gap import compute_generalization_gap
from visualisation.generalization_plots import plot_generalization_gap, plot_same_vs_lofo

compute_generalization_gap()
plot_generalization_gap()
plot_same_vs_lofo()
# %%
from evaluation.failure_analysis import retrieval_failure_analysis
failure_df = retrieval_failure_analysis()
print(failure_df.head())
# %%
from visualisation.hard_case_plots import plot_hard_cases_by_family, plot_hard_cases_by_length

plot_hard_cases_by_family()
plot_hard_cases_by_length()
# %%
from visualisation.lofo_plots import plot_lofo_f1
plot_lofo_f1()
# %% [markdown]
# # Run umap embedding
# %%
from experiments.run_embedding_umap import main
main()
# %%
from evaluation.generalization_gap import compute_generalization_gap
from visualisation.generalization_plots import plot_generalization_gap

compute_generalization_gap()
plot_generalization_gap()
# %% [markdown]
# Run DSR Ablation
# %%
from experiments.run_dsr_ablation import main
main()
# %%
from visualisation.dsr_plots import plot_dsr_comparison
plot_dsr_comparison()
# %%
from visualisation.dsr_retrieval_plots import plot_neighbor_quality
plot_neighbor_quality()
# %% [markdown]
# # Run DSR global Ablation
# %%
from experiments.run_dsr_global_ablation import main
main()
# %%
from experiments.run_dsr_global_all import main
main()
# %%
from visualisation.dsr_all_plots import (
    plot_dsr_all_vs_faiss_k14,
    plot_dsr_all_gain_distribution,
    plot_dsr_all_gain_vs_length,
)

plot_dsr_all_vs_faiss_k14()
plot_dsr_all_gain_distribution()
plot_dsr_all_gain_vs_length()
# %%
from visualisation.dsr_global_plots import plot_dsr_global_comparison
plot_dsr_global_comparison()
# %%
from visualisation.dsr_global_plots import plot_dsr_gain_vs_length
plot_dsr_gain_vs_length()
# %%
from experiments.run_dsr_faiss_hybrid import main
main()
# %%
from visualisation.dsr_hybrid_plots import (
    plot_dsr_hybrid_comparison,
    plot_dsr_hybrid_gain_distribution,
)

plot_dsr_hybrid_comparison()
plot_dsr_hybrid_gain_distribution()
# %%
from visualisation.final_model_comparison import plot_final_model_comparison

plot_final_model_comparison()
# %%
from visualisation.embedding_family_umap import plot_embedding_family_umap
plot_embedding_family_umap()
# %%
from visualisation.dsr_cluster_purity import plot_dsr_cluster_purity_heatmap
plot_dsr_cluster_purity_heatmap()
# %%
from visualisation.retrieval_family_confusion import plot_retrieval_family_confusion
plot_retrieval_family_confusion()
# %%
from visualisation.embedding_family_umap import plot_embedding_family_umap
from visualisation.dsr_cluster_purity import plot_dsr_cluster_purity_heatmap
from visualisation.retrieval_family_confusion import plot_retrieval_family_confusion

plot_embedding_family_umap()
plot_dsr_cluster_purity_heatmap()
plot_retrieval_family_confusion()
# %%
from visualisation.hybrid_family_confusion import plot_hybrid_family_confusion

plot_hybrid_family_confusion()
# %%
from experiments.generate_case_studies import main
main()
# %%
from visualisation.gain_vs_length import plot_gain_vs_length

plot_gain_vs_length()
# %%
import importlib

import visualisation.rna_radar_plot as rrp
import visualisation.rna_representation_matrix as rrm
import visualisation.rna_retrieval_graph as rrg
import visualisation.rna_tsne_embeddings as rte
import visualisation.similarity_vs_accuracy as sva

importlib.reload(rrp)
importlib.reload(rrm)
importlib.reload(rrg)
importlib.reload(rte)
importlib.reload(sva)
# %%
from data.build_dataset import build_dataset
from visualisation.rna_representation_matrix import (
    plot_rna_representation_matrix,
    plot_block_diagonal_rna_matrix,
)

dataset = list(build_dataset())

plot_rna_representation_matrix(
    z_path="results/dsr_global_Z.pt",
    save_path="results/plots/rna_representation_matrix.png"
)

plot_block_diagonal_rna_matrix(
    z_path="results/dsr_global_Z.pt",
    dataset=dataset,
    save_path="results/plots/rna_block_diagonal_matrix.png"
)
# %%
from visualisation.rna_tsne_embeddings import plot_rna_tsne
from data.build_dataset import build_dataset

dataset = list(build_dataset())

plot_rna_tsne(
    emb_path="results/kb_embeddings_rnafm_len256_full.pt",
    dataset=dataset,
)
# %%
from visualisation.similarity_vs_accuracy import plot_similarity_vs_accuracy

plot_similarity_vs_accuracy("results/ablation_results.csv")
# %%
from visualisation.rna_radar_plot import plot_radar
import pandas as pd

ab = pd.read_csv("results/ablation_results.csv")
dsr = pd.read_csv("results/dsr_global_all_results.csv")
hyb = pd.read_csv("results/dsr_faiss_hybrid_results.csv")
bench = pd.read_csv("results/benchmark_main.csv")

vienna = bench[bench["method"] == "vienna_only"]
faiss = ab[ab["method"] == "rnafm_faiss_guided_k14"]
dsr_only = dsr[dsr["method"] == "rnafm_dsr_global_guided_k14"]
hybrid = hyb[hyb["method"] == "rnafm_dsr_faiss_guided_k14"]

metrics = {
    "F1": hybrid["f1"].mean(),
    "Sensitivity": hybrid["sensitivity"].mean(),
    "PPV": hybrid["ppv"].mean(),
    "Purity": 0.85,   # replace with actual purity if saved
    "Speed": 0.80,    # normalized placeholder if you want
}

plot_radar(metrics)
# %%
import importlib
import visualisation.rna_retrieval_graph as rrg
importlib.reload(rrg)

import torch
from data.build_dataset import build_dataset

dataset = list(build_dataset())
families = [d["family"] for d in dataset]
Z = torch.load("results/dsr_global_Z.pt", map_location="cpu").numpy()

rrg.plot_rna_retrieval_graph(Z, families, n=120, k=4)
# %%
import pandas as pd

for path in [
    "results/benchmark_main.csv",
    "results/ablation_results.csv",
    "results/dsr_global_all_results.csv",
    "results/dsr_faiss_hybrid_results.csv",
]:
    df = pd.read_csv(path)
    print(path)
    print(df.columns.tolist())
    print()
# %%
import importlib
import visualisation.side_by_side_case_viz as ssv
importlib.reload(ssv)

ssv.plot_side_by_side_case(
    query_id="5S_rRNA-Bacteria-B00731",
    save_path="results/plots/case_studies/side_by_side_best.png"
)
# %%
from experiments.run_dsr_global_all import main as main1
from experiments.run_dsr_faiss_hybrid import main as main2
from experiments.run_ablation import main as main3
from experiments.run_benchmark import main as main4

main1()
main2()
main3()
main4()
# %%
from visualisation.side_by_side_case_viz import plot_side_by_side_case

plot_side_by_side_case(
    query_id="5S_rRNA-Bacteria-B00731",
    save_path="results/plots/case_studies/side_by_side_best.png"
)
# %%
ssv.plot_side_by_side_case(
    query_id="SRP-long_bacterial-Clos.ther._CP000568",
    save_path="results/plots/case_studies/side_by_side_typical.png"
)

ssv.plot_side_by_side_case(
    query_id="tRNA-tdbD00009470",
    save_path="results/plots/case_studies/side_by_side_worst.png"
)
# %%
import torch
import umap
import matplotlib.pyplot as plt
from data.build_dataset import build_dataset

dataset = list(build_dataset())
families = [x["family"] for x in dataset]

X = torch.load("results/kb_embeddings_rnafm_len256_full.pt").numpy()

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1)
X2 = reducer.fit_transform(X)

plt.figure(figsize=(8,6))

family_to_color = {f:i for i,f in enumerate(sorted(set(families)))}
colors = [family_to_color[f] for f in families]

plt.scatter(X2[:,0], X2[:,1], c=colors, cmap="tab10", s=12)

plt.title("RNA-FM embedding space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()
# %%
import torch
import matplotlib.pyplot as plt

Z = torch.load("results/dsr_global_Z.pt").cpu().numpy()

plt.figure(figsize=(6,6))
plt.imshow(Z, cmap="viridis")
plt.colorbar()
plt.title("Deep Self-Representation Matrix")
plt.show()
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

sns.scatterplot(
    data=df,
    x="avg_similarity",
    y="f1",
    hue="family"
)

plt.title("Neighbor Similarity vs Performance")
plt.show()
# %%
reducer = umap.UMAP(n_components=3)
X3 = reducer.fit_transform(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X3[:,0],
    X3[:,1],
    X3[:,2],
    c=colors,
    cmap="tab10",
    s=10
)

plt.title("3D RNA-FM embedding space")
plt.show()
# %%
import torch
import umap
import matplotlib.pyplot as plt
from data.build_dataset import build_dataset
import numpy as np

dataset = list(build_dataset())
families = [x["family"] for x in dataset]

X = torch.load("results/kb_embeddings_rnafm_len256_full.pt").numpy()

reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1)
X3 = reducer.fit_transform(X)

unique_fams = sorted(set(families))
fam_to_color = {f:i for i,f in enumerate(unique_fams)}
colors = [fam_to_color[f] for f in families]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X3[:,0],
    X3[:,1],
    X3[:,2],
    c=colors,
    cmap="tab10",
    s=12,
    alpha=0.9
)

ax.set_title("3D RNA-FM Embedding Space")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")

plt.show()
# %%
import pandas as pd

results = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

f1 = results["f1"].values

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X3[:len(f1),0],
    X3[:len(f1),1],
    X3[:len(f1),2],
    c=f1,
    cmap="plasma",
    s=12
)

fig.colorbar(sc, label="F1 score")

ax.set_title("3D Embedding Colored by Folding Accuracy")

plt.show()
# %%
lengths = [len(x["sequence"]) for x in dataset]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X3[:,0],
    X3[:,1],
    X3[:,2],
    c=lengths,
    cmap="viridis",
    s=10
)

fig.colorbar(sc, label="RNA Length")

ax.set_title("3D Embedding Colored by RNA Length")

plt.show()
# %%
import random

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=colors, cmap="tab10", s=8, alpha=0.4)

for _ in range(20):

    i = random.randint(0, len(X3)-1)
    j = random.randint(0, len(X3)-1)

    ax.plot(
        [X3[i,0], X3[j,0]],
        [X3[i,1], X3[j,1]],
        [X3[i,2], X3[j,2]],
        color="gray",
        alpha=0.2
    )

ax.set_title("RNA Retrieval Connections in Embedding Space")

plt.show()
# %%
Z = torch.load("results/dsr_latent_embeddings.pt").numpy()

reducer = umap.UMAP(n_components=3)
Z3 = reducer.fit_transform(Z)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    Z3[:,0],
    Z3[:,1],
    Z3[:,2],
    c=colors,
    cmap="tab10",
    s=12
)

ax.set_title("3D DSR Latent Space")

plt.show()
# %%
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "x": X3[:,0],
    "y": X3[:,1],
    "z": X3[:,2],
    "family": families
})

fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="family",
)

fig.show()
# %%
from visualisation.rna_3d_retrieval_flow import plot_3d_retrieval_flow

plot_3d_retrieval_flow(
    query_idx=0,
    k=5,
    save_path="results/plots/rna_3d_retrieval_flow_query0.png"
)
# %%
from visualisation.rna_3d_retrieval_flow import plot_3d_retrieval_flow

plot_3d_retrieval_flow(
    query_idx=0,
    k=5,
    save_path="results/plots/rna_3d_retrieval_flow_query0.png"
)
# %%
from data.build_dataset import build_dataset

dataset = list(build_dataset())

for i, row in enumerate(dataset):
    if row["family"] == "group_I_intron":
        print(i, row["id"], row["family"])
        break
# %%
from visualisation.rna_3panel_retrieval_flow import plot_3panel_retrieval_flow

plot_3panel_retrieval_flow(
    query_idx=0,
    k=5,
    save_path="results/plots/rna_3panel_retrieval_flow_query0.png"
)
# %%
from data.build_dataset import build_dataset
from visualisation.rna_3panel_retrieval_flow import plot_3panel_retrieval_flow

dataset = list(build_dataset())

for i, row in enumerate(dataset):
    if row["family"] == "5S_rRNA":
        print(i, row["id"], row["family"])
        break

plot_3panel_retrieval_flow(
    query_idx=i,
    k=5,
    save_path="results/plots/rna_3panel_retrieval_flow_5S.png"
)
# %%
from data.build_dataset import build_dataset
from visualisation.rna_3panel_retrieval_flow import plot_3panel_retrieval_flow

dataset = list(build_dataset())

for i, row in enumerate(dataset):
    if row["family"] == "tRNA":
        print(i, row["id"], row["family"])
        break

plot_3panel_retrieval_flow(
    query_idx=i,
    k=5,
    save_path="results/plots/rna_3panel_retrieval_flow_trna.png"
)
# %%
from data.build_dataset import build_dataset

dataset = list(build_dataset())

target_id = "5S_rRNA-Bacteria-B00731"

query_idx = None
for i, row in enumerate(dataset):
    if row["id"] == target_id:
        query_idx = i
        break

print("query_idx =", query_idx)
print("matched row =", dataset[query_idx] if query_idx is not None else "not found")
# %%
from data.build_dataset import build_dataset
import pandas as pd

dataset = list(build_dataset())
hyb = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

target_query_id = "5S_rRNA-Bacteria-B00731"   # or whatever appears in your CSV
target_family = "5S_rRNA"

# first try exact match
query_idx = None
for i, row in enumerate(dataset):
    if str(row["id"]) == str(target_query_id):
        query_idx = i
        break

# fallback: first row from same family
if query_idx is None:
    for i, row in enumerate(dataset):
        if row["family"] == target_family:
            query_idx = i
            break

print("query_idx =", query_idx)
print("matched row =", dataset[query_idx] if query_idx is not None else "not found")
# %%
from visualisation.rna_2panel_retrieval_flow import plot_2panel_retrieval_flow

plot_2panel_retrieval_flow(
    query_idx=query_idx,
    k=5,
    save_path="results/plots/case_studies/retrieval_flow_best.png"
)
# %%
from data.build_dataset import build_dataset
import pandas as pd

dataset = list(build_dataset())
hyb = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

target_query_id = "SRP-long_bacterial-Clos.ther._CP000568"   # or whatever appears in your CSV
target_family = "SRP"

# first try exact match
query_idx = None
for i, row in enumerate(dataset):
    if str(row["id"]) == str(target_query_id):
        query_idx = i
        break

# fallback: first row from same family
if query_idx is None:
    for i, row in enumerate(dataset):
        if row["family"] == target_family:
            query_idx = i
            break

print("query_idx =", query_idx)
print("matched row =", dataset[query_idx] if query_idx is not None else "not found")
# %%
from data.build_dataset import build_dataset
import pandas as pd

dataset = list(build_dataset())
hyb = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

target_query_id = "tRNA-tdbD00009470" # or whatever appears in your CSV
target_family = "tRNA"

# first try exact match
query_idx = None
for i, row in enumerate(dataset):
    if str(row["id"]) == str(target_query_id):
        query_idx = i
        break

# fallback: first row from same family
if query_idx is None:
    for i, row in enumerate(dataset):
        if row["family"] == target_family:
            query_idx = i
            break

print("query_idx =", query_idx)
print("matched row =", dataset[query_idx] if query_idx is not None else "not found")
# %%
from visualisation.rna_2panel_retrieval_flow import plot_2panel_retrieval_flow

plot_2panel_retrieval_flow(query_idx=200, k=5,
    save_path="results/plots/case_studies/retrieval_flow_best.png")

plot_2panel_retrieval_flow(query_idx=600, k=5,
    save_path="results/plots/case_studies/retrieval_flow_typical.png")

plot_2panel_retrieval_flow(query_idx=1000, k=5,
    save_path="results/plots/case_studies/retrieval_flow_worst.png")
# %%
from visualisation.family_benchmark_panel import plot_family_benchmark_panel_ordered

plot_family_benchmark_panel_ordered(
    csv_path="results/ablation_results.csv",
    exclude_families=["RfamFamily"],   # optional
    save_path="results/plots/family_benchmark_panel.png",
)
# %%
import pandas as pd

bench = pd.read_csv("results/benchmark_main.csv")
ab = pd.read_csv("results/ablation_results.csv")
dsr = pd.read_csv("results/dsr_global_all_results.csv")
hyb = pd.read_csv("results/dsr_faiss_hybrid_results.csv")

vienna = bench[bench["method"] == "vienna_only"].copy()
faiss = ab[ab["method"] == "rnafm_faiss_guided_k14"].copy()
dsr = dsr[dsr["method"] == "rnafm_dsr_global_guided_k14"].copy()
hyb = hyb[hyb["method"] == "rnafm_dsr_faiss_guided_k14"].copy()

combined = pd.concat([vienna, faiss, dsr, hyb], ignore_index=True)
combined.to_csv("results/final_family_benchmark_methods.csv", index=False)

print(combined["method"].value_counts())
# %%
#from visualisation.family_benchmark_panel import plot_family_benchmark_panel_compact

#plot_family_benchmark_panel_compact(
#    csv_path="results/final_family_benchmark_methods.csv",
#    exclude_families=["RfamFamily"],   # optional
#    save_path="results/plots/final_family_benchmark_panel.png",
#)
# %%
import importlib
import visualisation.family_benchmark_panel as fb

importlib.reload(fb)
fb.plot_family_benchmark_panel_ordered(
    csv_path="results/final_family_benchmark_methods.csv",
    order_by="vienna_only",
    exclude_families=["RfamFamily"],
    save_path="results/plots/final_family_benchmark_panel_ordered.png",
)

fb.plot_family_gain_panel(
    csv_path="results/final_family_benchmark_methods.csv",
    exclude_families=["RfamFamily"],
    save_path="results/plots/family_gain_panel.png",
)
# %%
from visualisation.final_eccb_figure import plot_final_eccb_figure
plot_final_eccb_figure()
# %%
import importlib
import visualisation.length_benchmark_panel as lbp
importlib.reload(lbp)

lbp.plot_length_benchmark_panel_ordered(
    csv_path="results/final_family_benchmark_methods.csv",
    order_by="vienna_only",
    save_path="results/plots/final_length_benchmark_panel_ordered.png",
)

lbp.plot_length_gain_panel(
    csv_path="results/final_family_benchmark_methods.csv",
    save_path="results/plots/length_gain_panel.png",
)
# %%
import importlib
import visualisation.side_by_side_case_viz as ssv
importlib.reload(ssv)

ssv.plot_side_by_side_case(
    query_id="5S_rRNA-Bacteria-B00731",
    save_path="results/plots/case_studies/side_by_side_best.png"
)

ssv.plot_side_by_side_case(
    query_id="SRP-long_bacterial-Clos.ther._CP000568",
    save_path="results/plots/case_studies/side_by_side_typical.png"
)

ssv.plot_side_by_side_case(
    query_id="tRNA-tdbD00009470",
    save_path="results/plots/case_studies/side_by_side_worst.png"
)
# %%
from evaluation.family_transfer import compute_family_transfer
from visualisation.family_transfer_plots import (
    plot_family_transfer_gain,
    plot_family_transfer_scatter,
)

df_transfer = compute_family_transfer()
print(df_transfer)

plot_family_transfer_gain()
plot_family_transfer_scatter()
# %%
import importlib
import visualisation.dsr_global_plots as dsrplots
importlib.reload(dsrplots)
# %%
from visualisation.dsr_global_plots import plot_dsr_gain_distribution
plot_dsr_gain_distribution()
# %%

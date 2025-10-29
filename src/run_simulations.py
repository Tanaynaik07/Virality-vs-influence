from src.preprocess import load_graph, select_seeds
from src.ic_model import independent_cascade
from src.lt_model import linear_threshold
from src.hybrid_model import hybrid_model
from src.visualize import plot_results
from src.influence_maximization import celf_im
from src.cluster_seeding import cluster_based_seeds
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
 

def plot_line_and_heatmap(df, dataset_name="Dataset"):
    """
    df: DataFrame from experiments
    Columns needed: 'Seeds (k)', 'IC prob (p)', 'Hybrid prob (q)', 'Cascade IC', 'Cascade LT', 'Cascade Hybrid'
    """

    # -------------------
    # 1. Line Plot: Cascade vs IC probability
    # -------------------
    plt.figure(figsize=(10, 6))
    for model in ["Cascade IC", "Cascade LT", "Cascade Hybrid"]:
        mean_df = df.groupby("IC prob (p)")[model].mean()
        plt.plot(mean_df.index, mean_df.values, marker='o', label=model)
    plt.title(f"{dataset_name}: Influence Spread vs IC Probability")
    plt.xlabel("IC Probability (p)")
    plt.ylabel("Average Cascade Size")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_lineplot.png")
    plt.show()

    # -------------------
    # 2. Heatmap: Hybrid cascade size over p and q
    # -------------------
    pivot = df.pivot_table(index="Hybrid prob (q)", columns="IC prob (p)", values="Cascade Hybrid", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title(f"{dataset_name}: Hybrid Cascade Size Heatmap")
    plt.xlabel("IC Probability (p)")
    plt.ylabel("Hybrid External Influence (q)")
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_heatmap.png")
    plt.show()

def run_all(path="data/twitter_combined.txt.gz"):
    """
    Quick demo run: picks 5 seed nodes, runs IC, LT, and Hybrid,
    prints & plots the result.
    """
    print("Loading dataset...")
    G = load_graph(path)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    seeds = select_seeds(G, k=5, method="degree")
    print("Selected seeds:", seeds)
    
    ic = independent_cascade(G, seeds, p=0.1, steps=10)
    lt = linear_threshold(G, seeds, steps=10)
    hy = hybrid_model(G, seeds, p=0.1, q=0.02, steps=10)
    
    results = {
        "IC": len(ic),
        "LT": len(lt),
        "Hybrid": len(hy),
    }
    print("Results:", results)
    
    plot_results(results)
    return results

def experiment(path="data/twitter_combined.txt.gz"):
    print(f"Loading dataset for experiments: {path}")
    G_full = load_graph(path)

    # --- take a smaller subgraph for faster runtime ---
    nodes_subset = list(G_full.nodes())[:5000]
    G = G_full.subgraph(nodes_subset).copy()

    results_table = []

    # Sweep k from 5 to 100 in steps of 5
    for k in range(5, 51, 5):
        for p in [0.05]:   # you can expand this to more p values if needed
            for q in [0.01]:  # same for q values
                seeds_degree = select_seeds(G, k=k, method="degree")
                seeds_cluster = cluster_based_seeds(G, k=k, method="louvain")
                seeds_celf, _ = celf_im(G, k=k, p=p, steps=5)

                strategies = {
                    "Degree": seeds_degree,
                    "Cluster": seeds_cluster,
                    "CELF": seeds_celf,
                }

                for strat_name, seeds in strategies.items():
                    ic = independent_cascade(G, seeds, p=p, steps=5)
                    lt = linear_threshold(G, seeds, steps=5)
                    hy = hybrid_model(G, seeds, p=p, q=q, steps=5)

                    results_table.append({
                        "Dataset": os.path.basename(path),
                        "Strategy": strat_name,
                        "Seeds (k)": k,
                        "IC prob (p)": p,
                        "Hybrid prob (q)": q,
                        "Cascade IC": len(ic),
                        "Cascade LT": len(lt),
                        "Cascade Hybrid": len(hy)
                    })

    df = pd.DataFrame(results_table)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/experiment_{os.path.basename(path)}.csv", index=False)
    print(f"✅ Experiment saved for {path}")
    return df



if __name__ == "__main__":
    # quick run
    results = run_all("data/twitter_combined.txt.gz")
    print("Final Results:", results)

    # full experiments on multiple datasets
    datasets = ["data/twitter_combined.txt.gz", "data/facebook_combined.txt.gz"]
    all_results = []
    for dataset in datasets:
        df = experiment(dataset)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("results/experiment_all_datasetss.csv", index=False)
    print("✅ All datasets experiment saved to results/experiment_all_datasets.csv")

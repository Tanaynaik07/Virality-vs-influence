from src.preprocess import load_graph, select_seeds
from src.ic_model import independent_cascade
from src.lt_model import linear_threshold
from src.hybrid_model import hybrid_model
from src.visualize import plot_results
import random
 
from src.cluster_seeding import cluster_based_seeds
import pandas as pd
import matplotlib.pyplot as plt
import os
def greedy_im(G, k=5, p=0.1, steps=10):
    """
    Greedy Influence Maximization:
    Iteratively add the node that maximizes marginal gain.
    """
    seeds = []
    spread = []

    for _ in range(k):
        best_node, best_spread = None, -1
        for node in G.nodes():
            if node in seeds:
                continue
            # test adding node to seeds
            temp_seeds = seeds + [node]
            activated = independent_cascade(G, temp_seeds, p=p, steps=steps)
            if len(activated) > best_spread:
                best_spread = len(activated)
                best_node = node
        seeds.append(best_node)
        spread.append(best_spread)
    return seeds, spread[-1]


def celf_im(G, k=5, p=0.1, steps=10):
    """
    CELF (Cost-Effective Lazy Forward):
    Optimized greedy IM using priority queue.
    Much faster than naive greedy.
    """
    # initial marginal gains
    marg_gain = []
    for node in G.nodes():
        activated = independent_cascade(G, [node], p=p, steps=steps)
        marg_gain.append((len(activated), node, 0))  # (gain, node, last_updated)

    # sort by gain
    marg_gain.sort(reverse=True)
    seeds = []
    spread = []

    for i in range(k):
        found = False
        while not found:
            gain, node, last = marg_gain[0]
            if last == i:  # still valid
                seeds.append(node)
                spread.append(gain)
                marg_gain.pop(0)
                found = True
            else:  # recompute
                activated = independent_cascade(G, seeds + [node], p=p, steps=steps)
                new_gain = len(activated) - (spread[-1] if spread else 0)
                marg_gain[0] = (new_gain, node, i)
                marg_gain.sort(reverse=True)

    return seeds, spread[-1]

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
    """
    Systematic experiments: runs multiple configurations with
    varying seeds (k), IC probability (p), and Hybrid probability (q).
    Saves results to CSV + Excel and generates a comparison chart.
    """
    print("Loading dataset for experiments...")
    G = load_graph(path)
    results_table = []

    # vary k, IC probability p, and hybrid q
    for k in [5, 10, 20]:
        for p in [0.05, 0.1, 0.2]:
            for q in [0.01, 0.02]:
                seeds = select_seeds(G, k=k, method="degree")
                
                ic = independent_cascade(G, seeds, p=p, steps=10)
                lt = linear_threshold(G, seeds, steps=10)
                hy = hybrid_model(G, seeds, p=p, q=q, steps=10)
                
                results_table.append({
                    "Seeds (k)": k,
                    "IC prob (p)": p,
                    "Hybrid prob (q)": q,
                    "Cascade IC": len(ic),
                    "Cascade LT": len(lt),
                    "Cascade Hybrid": len(hy)
                })

    df = pd.DataFrame(results_table)

    # save results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/experiment_results.csv"
    xlsx_path = "results/experiment_results.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"✅ Experiment results saved to:\n- {csv_path}\n- {xlsx_path}")

    # plot a grouped bar chart
    df_plot = df.melt(
        id_vars=["Seeds (k)", "IC prob (p)", "Hybrid prob (q)"],
        value_vars=["Cascade IC", "Cascade LT", "Cascade Hybrid"],
        var_name="Model", value_name="Spread"
    )

    plt.figure(figsize=(12, 6))
    for key, grp in df_plot.groupby(["Seeds (k)", "IC prob (p)"]):
        grp.plot.bar(
            x="Model", y="Spread", ax=plt.gca(),
            label=f"k={key[0]}, p={key[1]}"
        )
    plt.title("Influence Spread Comparison Across Models")
    plt.ylabel("Number of Activated Nodes")
    plt.xlabel("Model")
    plt.legend()
    plt.tight_layout()

    plot_path = "results/experiment_plot.png"
    plt.savefig(plot_path)
    print(f"📊 Comparison plot saved to {plot_path}")

    return df


if __name__ == "__main__":
    # quick run
    results = run_all("data/twitter_combined.txt.gz")
    print("Final Results:", results)

    # full experiments
    df = experiment("data/twitter_combined.txt.gz")
    print(df.head())

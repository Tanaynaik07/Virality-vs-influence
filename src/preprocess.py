import networkx as nx
import gzip

def load_graph(path: str):
    """
    Load Twitter ego network dataset into NetworkX graph.
    """
    G = nx.Graph()
    with gzip.open(path, "rt") as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(int(u), int(v))
    return G

def select_seeds(G, k=5, method="degree"):
    """
    Select seed nodes for diffusion simulation.
    - degree: top k highest degree nodes
    - pagerank: top k nodes by PageRank
    - random: random k nodes
    """
    if method == "degree":
        seeds = sorted(G.degree, key=lambda x: x[1], reverse=True)[:k]
        return [node for node, _ in seeds]
    elif method == "pagerank":
        pr = nx.pagerank(G)
        seeds = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]
        return [node for node, _ in seeds]
    else:
        import random
        return random.sample(list(G.nodes()), k)

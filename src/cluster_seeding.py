import networkx as nx
import random

def cluster_based_seeds(G, k=5, method="louvain"):
    """
    Selects seeds from different clusters/communities to maximize coverage.
    Requires python-louvain if method="louvain".
    """
    try:
        import community  # python-louvain package
    except ImportError:
        raise ImportError("Please install python-louvain: pip install python-louvain")

    if method == "louvain":
        partition = community.best_partition(G)
    else:
        raise ValueError("Only Louvain clustering is supported for now")

    # group nodes by community
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    # pick seeds from largest communities
    sorted_comms = sorted(communities.values(), key=len, reverse=True)

    seeds = []
    i = 0
    while len(seeds) < k and i < len(sorted_comms):
        # pick one representative from each community
        comm_nodes = sorted_comms[i]
        seeds.append(random.choice(comm_nodes))
        i += 1

    # if not enough, fill randomly
    while len(seeds) < k:
        seeds.append(random.choice(list(G.nodes())))

    return seeds

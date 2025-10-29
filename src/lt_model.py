import random

def linear_threshold(G, seeds, steps=10):
    """
    Linear Threshold Model simulation.
    Each node has random threshold between 0-1.
    """
    thresholds = {node: random.random() for node in G.nodes()}
    active = set(seeds)

    for _ in range(steps):
        new_active = set()
        for node in G.nodes():
            if node not in active:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue
                active_neighbors = len([n for n in neighbors if n in active])
                if (active_neighbors / len(neighbors)) >= thresholds[node]:
                    new_active.add(node)
        if not new_active:
            break
        active |= new_active
    return active

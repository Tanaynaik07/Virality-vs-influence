import random

def independent_cascade(G, seeds, p=0.1, steps=10):
    """
    Independent Cascade Model simulation.
    G: Graph
    seeds: list of seed nodes
    p: probability of influence
    steps: max steps
    """
    active = set(seeds)
    newly_active = set(seeds)
    
    for _ in range(steps):
        next_newly_active = set()
        for node in newly_active:
            for neighbor in G.neighbors(node):
                if neighbor not in active and random.random() <= p:
                    next_newly_active.add(neighbor)
        if not next_newly_active:
            break
        active |= next_newly_active
        newly_active = next_newly_active
    return active

import random
from .ic_model import independent_cascade

def hybrid_model(G, seeds, p=0.1, q=0.05, steps=10):
    """
    Hybrid Model: IC + external influence.
    - IC: normal spread
    - External: each inactive node has probability q to be activated externally
    """
    active = set(seeds)
    newly_active = set(seeds)
    
    for _ in range(steps):
        next_newly_active = set()
        
        # Internal influence (IC)
        for node in newly_active:
            for neighbor in G.neighbors(node):
                if neighbor not in active and random.random() <= p:
                    next_newly_active.add(neighbor)
        
        # External influence
        for node in G.nodes():
            if node not in active and random.random() <= q:
                next_newly_active.add(node)

        if not next_newly_active:
            break
        active |= next_newly_active
        newly_active = next_newly_active
    return active

# src/env.py
import numpy as np
import networkx as nx

def detection_probability(G: nx.Graph, H, j: int, alpha: float, radius: int) -> float:
    neighbors = []
    for h in H:
        try:
            d = nx.shortest_path_length(G, h, j)
        except nx.NetworkXNoPath:
            continue
        if d <= radius:
            neighbors.append(h)
    if not neighbors:
        return 0.0
    p = 1.0
    for _ in neighbors:
        p *= (1.0 - alpha)
    return 1.0 - p

def simulate_round(G, H, j, node_values, alpha: float, radius: int,
                   reward_detect: float, rng) -> tuple[float, float]:
    """
    H: honeypot nodes
    j: attacked node
    node_values: dict[node] -> value (10 for asset, 3 for normal, etc.)
    Returns (attacker_reward, defender_reward)
    """
    p_det = detection_probability(G, H, j, alpha=alpha, radius=radius)
    detected = rng.random() < p_det

    if detected:
        r_att = -reward_detect
        r_def = reward_detect
    else:
        v = node_values[j]
        r_att = v
        r_def = -v

    return r_att, r_def

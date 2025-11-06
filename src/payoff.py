from typing import Sequence, List
import numpy as np
import networkx as nx


def p_det_direct(target: int, H: Sequence[int]) -> float:
    return 1.0 if target in H else 0.0


def p_det_neighborhood(G: nx.Graph, target: int, H: Sequence[int], r: int, alpha: float) -> float:
    nodes = {target}
    frontier = {target}
    for _ in range(r):
        nxt = set()
        for u in frontier:
            nxt.update(G.neighbors(u))
        nodes.update(nxt)
        frontier = nxt
    in_range = nodes.intersection(set(H))
    p_miss = 1.0
    for _ in in_range:
        p_miss *= (1.0 - alpha)
    return 1.0 - p_miss


def build_payoff_matrix(G: nx.Graph, 
                        actions: List[Sequence[int]], 
                        node_values: dict, 
                        reward_detect: float, 
                        model: str = 'neighborhood', 
                        radius: int = 1, 
                        alpha: float = 0.7) -> np.ndarray:
    n_nodes = G.number_of_nodes()
    A = np.zeros((len(actions), n_nodes), dtype=float)
    for i, H in enumerate(actions):
        for j in range(n_nodes):
            if model == 'direct':
                p = p_det_direct(j, H)
            else:
                p = p_det_neighborhood(G, j, H, radius, alpha)
            L = node_values[j]
            A[i, j] = p * reward_detect - (1.0 - p) * L
    return A
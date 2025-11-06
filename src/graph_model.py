from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np


@dataclass
class GraphState:
    G: nx.Graph
    node_roles: Dict[int, str]
    node_values: Dict[int, float]
    candidates: List[int]
    assets: List[int]


def grid_graph(rows: int, cols: int) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    mapping = { (r, c): r*cols + c for r, c in G.nodes() }
    G = nx.relabel_nodes(G, mapping)
    return G


def build_graph(rows: int, cols: int, assets: List[int], candidate_honeypots: str | List[int],
    asset_value: float, normal_value: float) -> GraphState:
    G = grid_graph(rows, cols)
    n = G.number_of_nodes()
    roles = {i: 'normal' for i in range(n)}
    for a in assets:
        roles[a] = 'asset'
    if candidate_honeypots == 'all':
        candidates = list(G.nodes())
        candidates = [n for n in candidates if n not in assets]
    else:
        candidates = list(candidate_honeypots)
    values = {i: (asset_value if roles[i] == 'asset' else normal_value) for i in G.nodes()}
    return GraphState(G=G, node_roles=roles, node_values=values, candidates=candidates, assets=assets)
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Config:
    seed: int
    out_dir: str
    plots: bool
    graph_type: str
    rows: int
    cols: int
    assets: List[int]
    cand: str | List[int]
    asset_value: float
    normal_value: float
    B: int
    det_model: str
    radius: int
    alpha: float
    reward_detect: float
    action_mode: str
    max_actions: int
    method: str
    rounds: int
    eta_def: float
    eta_att: float
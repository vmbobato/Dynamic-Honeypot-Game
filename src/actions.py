from itertools import combinations
from typing import List, Sequence


def enumerate_actions(candidates: Sequence[int], B: int, max_actions: int | None = None) -> List[tuple]:
    comb_iter = combinations(candidates, B)
    acts = []
    for k, H in enumerate(comb_iter):
        acts.append(tuple(H))
        if max_actions and k+1 >= max_actions:
            break
    return acts
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_regret_like(pay_hist: np.ndarray, out_path: str):
    x = np.arange(1, len(pay_hist)+1)
    y = np.convolve(pay_hist, np.ones(100)/100, mode='same')
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('round')
    plt.ylabel('rolling mean payoff')
    plt.title('MWU convergence (rolling mean payoff)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_node_heat(G: nx.Graph, probs: Sequence[float], out_path: str, title: str = "Node probabilities"):
    probs = np.asarray(probs, dtype=float)
    pos = nx.spring_layout(G, seed=3)

    vmin = float(np.min(probs))
    vmax = float(np.max(probs))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:  # avoid zero-range normalize
        vmax = vmin + 1e-12

    plt.figure()
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=500,
        node_color=probs,
        cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos)
    cbar = plt.colorbar(nodes)
    cbar.set_label("probability")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_rolling_mean(pay_hist: np.ndarray, out_path: str, window: int = 300):
    x = np.arange(1, len(pay_hist)+1)
    k = max(1, int(window))
    kernel = np.ones(k) / k
    y = np.convolve(pay_hist, kernel, mode='same')
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('round')
    plt.ylabel('rolling mean payoff')
    plt.title(f'MWU convergence (rolling mean, window={k})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_cumulative_mean(pay_hist: np.ndarray, out_path: str):
    pay_hist = np.asarray(pay_hist, dtype=float)
    x = np.arange(1, len(pay_hist)+1)
    cum = np.cumsum(pay_hist) / x
    plt.figure()
    plt.plot(x, cum)
    plt.xlabel('round')
    plt.ylabel('cumulative mean payoff')
    plt.title('MWU convergence (cumulative mean)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
import argparse, yaml, os
import numpy as np
from pathlib import Path

from src.io_utils import ensure_dir
from src.graph_model import build_graph
from src.actions import enumerate_actions
from src.payoff import build_payoff_matrix
from src.mwu import MWUSolver
from src.attacker_bandit import SoftmaxBanditAttacker
from src.env import simulate_round
from src.visualize import *


def run_full_info_mode(cfg, Gs, actions, out_dir):
    """Original MWU equilibrium solver: both players know the payoff matrix A."""
    B = cfg['budget']['B']
    dcfg = cfg['detection']
    scfg = cfg['solver']

    # Build payoff matrix A[def_action, attack_node]
    A = build_payoff_matrix(
        Gs.G, actions, Gs.node_values, dcfg['reward_detect'],
        model=dcfg['model'],
        radius=dcfg['radius'],
        alpha=dcfg['alpha']
    )

    mwu = MWUSolver(A, eta_def=scfg['eta_def'], eta_att=scfg['eta_att'], seed=cfg['seed'])
    res = mwu.run(T=scfg['rounds'])
    x_bar, q_bar = res['x_bar'], res['q_bar']

    # defender node marginals: expected honeypots per node
    node_probs = np.zeros(Gs.G.number_of_nodes())
    for k, H in enumerate(actions):
        for node in H:
            node_probs[node] += x_bar[k]

    np.savetxt(Path(out_dir) / 'defender_mix_mwu.csv', x_bar, delimiter=',')
    np.savetxt(Path(out_dir) / 'attacker_mix_mwu.csv', q_bar, delimiter=',')
    np.savetxt(Path(out_dir) / 'defender_marginals.csv', node_probs, delimiter=',')
    np.savetxt(Path(out_dir) / 'pay_hist.csv', res['pay_hist'], delimiter=',')

    if cfg['output']['plots']:
        plot_rolling_mean(res['pay_hist'], str(Path(out_dir) / 'payoff_trend.png'), window=300)
        plot_cumulative_mean(res['pay_hist'], str(Path(out_dir) / 'payoff_cumulative.png'))
        plot_node_heat(
            Gs.G, node_probs,
            str(Path(out_dir) / 'placement_heatmap.png'),
            title="Honeypot Placement Heatmap (MWU, full info)"
        )
        plot_node_heat(
            Gs.G, q_bar,
            str(Path(out_dir) / 'attack_heatmap.png'),
            title="Node Attack Heatmap (MWU, full info)"
        )
    print('Saved full-info MWU results to', out_dir)


def run_bandit_attacker_mode(cfg, Gs, actions, out_dir):
    """
    Incomplete-information mode:
    - Defender: simple uniform random over honeypot placements (for now)
    - Attacker: Softmax bandit over nodes, does NOT know which nodes are assets.
    - Only sees scalar reward each round from simulate_round.
    """
    rng = np.random.default_rng(cfg['seed'])
    dcfg = cfg['detection']
    scfg = cfg['solver']
    T = scfg['rounds']
    B = cfg['budget']['B']

    n_def_actions = len(actions)
    def_probs = np.ones(n_def_actions, dtype=float) / n_def_actions  # uniform defender

    attacker = SoftmaxBanditAttacker(
        n_nodes=Gs.G.number_of_nodes(),
        eta=scfg.get('eta_att', 0.1),
        seed=cfg['seed']
    )

    pay_hist = []
    att_counts = np.zeros(Gs.G.number_of_nodes(), dtype=int)
    def_counts = np.zeros(Gs.G.number_of_nodes(), dtype=float)

    for t in range(T):
        # Defender: sample honeypot placement H_t
        H_idx = rng.choice(n_def_actions, p=def_probs)
        H = actions[H_idx]

        # Attacker: sample node j_t (no knowledge of assets)
        j = attacker.sample_node()

        # Environment: compute outcome + rewards
        r_att, r_def = simulate_round(
            Gs.G, H, j, Gs.node_values,
            alpha=dcfg['alpha'],
            radius=dcfg['radius'],
            reward_detect=dcfg['reward_detect'],
            rng=rng
        )
        pay_hist.append(r_att)

        # Attacker updates only from (j, r_att)
        attacker.update(j, r_att)

        # Stats
        att_counts[j] += 1
        for node in H:
            def_counts[node] += 1

    pay_hist = np.asarray(pay_hist, dtype=float)
    att_dist = att_counts / att_counts.sum()
    # approximate expected honeypots per node
    def_marginals = def_counts / def_counts.sum() * B

    np.savetxt(Path(out_dir) / 'attacker_mix_bandit.csv', att_dist, delimiter=',')
    np.savetxt(Path(out_dir) / 'defender_marginals_uniform.csv', def_marginals, delimiter=',')
    np.savetxt(Path(out_dir) / 'pay_hist_bandit.csv', pay_hist, delimiter=',')

    if cfg['output']['plots']:
        plot_cumulative_mean(
            pay_hist, str(Path(out_dir) / 'payoff_cumulative_bandit.png')
        )
        plot_node_heat(
            Gs.G, def_marginals,
            str(Path(out_dir) / 'placement_heatmap_bandit.png'),
            title="Honeypot Placement Heatmap (Uniform Defender)"
        )
        plot_node_heat(
            Gs.G, att_dist,
            str(Path(out_dir) / 'attack_heatmap_bandit.png'),
            title="Node Attack Heatmap (Bandit, Incomplete Info)"
        )
    print('Saved bandit-attacker results to', out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg['output']['dir']
    ensure_dir(out_dir)

    gcfg = cfg['graph']
    vcfg = cfg['values']
    Gs = build_graph(
        rows=gcfg['rows'],
        cols=gcfg['cols'],
        assets=gcfg['assets'],
        candidate_honeypots=gcfg['candidate_honeypots'],
        asset_value=vcfg['asset_value'],
        normal_value=vcfg['normal_value']
    )

    B = cfg['budget']['B']
    act_cfg = cfg['actions']
    actions = enumerate_actions(Gs.candidates, B, max_actions=act_cfg.get('max_actions', None))
    print(f"Defender actions: {len(actions)}")

    scfg = cfg['solver']
    mode = scfg.get('mode', 'full_info')

    if mode == 'full_info':
        run_full_info_mode(cfg, Gs, actions, out_dir)
    elif mode == 'bandit_attacker':
        run_bandit_attacker_mode(cfg, Gs, actions, out_dir)
    else:
        raise ValueError(f"Unknown solver mode: {mode}")


if __name__ == '__main__':
    main()

import argparse, yaml, os
import numpy as np
from pathlib import Path
from src.io_utils import ensure_dir
from src.graph_model import build_graph
from src.actions import enumerate_actions
from src.payoff import build_payoff_matrix
from src.mwu import MWUSolver
from src.visualize import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg['seed'])
    out_dir = cfg['output']['dir']
    ensure_dir(out_dir)
    gcfg = cfg['graph']
    vcfg = cfg['values']
    Gs = build_graph(rows=gcfg['rows'], cols=gcfg['cols'], assets=gcfg['assets'],
                     candidate_honeypots=gcfg['candidate_honeypots'],
                     asset_value=vcfg['asset_value'], normal_value=vcfg['normal_value']
                     )
    B = cfg['budget']['B']
    act_cfg = cfg['actions']
    actions = enumerate_actions(Gs.candidates, B, max_actions=act_cfg.get('max_actions', None))
    print(f"Defender actions: {len(actions)}")
    dcfg = cfg['detection']
    A = build_payoff_matrix(Gs.G, actions, Gs.node_values, dcfg['reward_detect'],
                            model=dcfg['model'],
                            radius=dcfg['radius'],
                            alpha=dcfg['alpha']
                            )
    scfg = cfg['solver']
    mwu = MWUSolver(A, eta_def=scfg['eta_def'], eta_att=scfg['eta_att'], seed=cfg['seed'])
    res = mwu.run(T=scfg['rounds'])
    x_bar, q_bar = res['x_bar'], res['q_bar']
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
        plot_node_heat(Gs.G, node_probs, str(Path(out_dir) / 'placement_heatmap.png'), title="Honeypot Placement Heatmap")
        plot_node_heat(Gs.G, q_bar, str(Path(out_dir) / 'attack_heatmap.png'), title="Node Attack Heatmap")
    print('Saved results to', out_dir)


if __name__ == '__main__':
    main()
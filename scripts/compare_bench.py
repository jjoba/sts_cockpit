'''
Compares two models in a bench folder written by interaction_bench.py (needs all sections). Run from the project root:

    python scripts/compare_bench.py bench=v22_all control=fnn_v21_ft_st treatment=fnn_v22_ft_st_cum

Writes report.md next to the bench output and prints it. The treatment is whatever changed (e.g. the cumulative upgrade encoding), the control is the same
recipe without it. "Rare" below means fewer than RARE training rows had that upgrade state (or combination of states).
'''

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = os.getcwd()
RARE = 5000
STRONG = 0.03 # a pair counts as having an interaction worth tracing into rare states above this (logit)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.float_format', lambda v: f'{v:.3f}')

N_COLS = 929 + 5 # features then the 5 head labels, Heart last (see interaction_bench.py)

def test_metrics(models):
    '''
    Heart-head metrics from the logits the bench cached for the held-out finetuning file. Precision, recall and F1 are at the standard 0.5 threshold.
    Best F1 (and its precision / recall) is at each model's own best threshold, chosen on this same file so read it across models, not as absolute,
    and recall at the top 5% / 10% of predictions needs no threshold.
    '''
    y = np.memmap(f'{ROOT}/data/training_data/finetuning/test/test.bin', dtype = np.int8, mode = 'r').reshape(-1, N_COLS)[:, -1].astype(np.int8)
    rows = []
    for m in models:
        path = f'{ROOT}/model_objects/production/{m}.pth'
        st = os.stat(path)
        f = f'{ROOT}/data/bench/cache/test_{m}_{st.st_size}_{int(st.st_mtime)}.npy'
        if not os.path.exists(f):
            continue
        p = np.clip(1 / (1 + np.exp(-np.load(f).astype(np.float64))), 1e-7, 1 - 1e-7)
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        k = f1.argmax()
        pos = p >= 0.5
        tp = (pos & (y == 1)).sum()
        pr, rc = tp / max(pos.sum(), 1), tp / y.sum()
        bins = np.array_split(np.argsort(p), 10) # equal frequency bins, as train_model.py's ece
        ece = sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in bins) / len(y)
        row = {'model': m, 'log_loss': -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), 'auc': roc_auc_score(y, p), 'ece': ece, 'brier': np.mean((p - y) ** 2),
               'avg_precision': average_precision_score(y, p), 'precision@0.5': pr, 'recall@0.5': rc, 'f1@0.5': 2 * pr * rc / max(pr + rc, 1e-12), 'n_pred_pos@0.5': int(pos.sum()),
               'best_f1': f1[k], 'precision@bestF1': prec[k], 'recall@bestF1': rec[k], 'threshold': thr[k]}
        order = np.argsort(-p)
        for q in (0.05, 0.10):
            row[f'recall top {int(q * 100)}%'] = y[order[:int(len(y) * q)]].sum() / y.sum()
        rows.append(row)
    return pd.DataFrame(rows)

def table(df, digits = 3):
    return '```\n' + df.to_string(index = False, float_format = lambda v: f'{v:.{digits}f}') + '\n```\n'

def main(bench, a, b):
    d = f'{ROOT}/model_objects/bench/{bench}'
    out = [f'# {b} vs {a}  (bench {bench})\n']

    # Held-out finetuning metrics from the experiments log
    exp = pd.read_csv(f'{ROOT}/model_objects/experiments.csv').set_index('model_version')
    cols = [c for c in ['best_ft_log_loss', 'best_ft_auc', 'best_ft_ece', 'best_ft_brier', 'best_ft_avg_precision', 'best_ft_best_f1'] if c in exp.columns]
    if a in exp.index and b in exp.index:
        t = exp.loc[[a, b], cols].reset_index()
        out += ['## Heart head, held-out finetuning set\n', table(t, 5)]

    tm = test_metrics([a, b])
    if len(tm):
        out += ['## Standard metrics on the held-out set (log loss, AUC, calibration error, Brier, precision, recall, F1 at 0.5)\n', table(tm, 5)]

    # 1. Upgrade effects
    eff = pd.read_csv(f'{d}/upgrade_effects.csv')
    ea, eb = eff[f'upgrade_effect_{a}'], eff[f'upgrade_effect_{b}']
    eff['rare'] = np.minimum(eff.n_train_plain, eff.n_train_upgraded) < RARE
    eff['delta'] = eb - ea
    rows = []
    for name, g in [('all cards', eff), (f'rare state (<{RARE} rows in plain or upgraded)', eff[eff.rare]), ('well supported', eff[~eff.rare])]:
        rows.append({'cards': name, 'n': len(g), f'mean effect {a}': g[f'upgrade_effect_{a}'].mean(), f'mean effect {b}': g[f'upgrade_effect_{b}'].mean(),
                     f'share <0 {a}': (g[f'upgrade_effect_{a}'] < 0).mean(), f'share <0 {b}': (g[f'upgrade_effect_{b}'] < 0).mean(),
                     'corr(effects)': g[[f'upgrade_effect_{a}', f'upgrade_effect_{b}']].corr().iloc[0, 1]})
    out += ['## Does upgrading look like an upgrade?\n',
            'Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.\n',
            table(pd.DataFrame(rows))]
    worst = eff.sort_values(f'upgrade_effect_{b}').head(8)[['card', 'char', 'n_train_plain', 'n_train_upgraded', f'upgrade_effect_{a}', f'upgrade_effect_{b}']]
    moved = eff.reindex(eff.delta.abs().sort_values(ascending = False).index).head(8)[['card', 'char', 'n_train_plain', 'n_train_upgraded', f'upgrade_effect_{a}', f'upgrade_effect_{b}']]
    out += [f'Lowest upgrade effects under {b}:\n', table(worst), f'Biggest changes {a} to {b}:\n', table(moved)]

    # 2. Rare state calibration
    rare = pd.read_csv(f'{d}/rare_cells.csv')
    bins = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 10 ** 12)]
    rows = []
    for lo, hi in bins:
        row = {'train rows of state': f'{lo}-{hi if hi < 10 ** 11 else "inf"}'}
        for m in (a, b):
            g = rare[(rare.model == m) & (rare.n_train >= lo) & (rare.n_train < hi)]
            row[f'cells'] = len(g)
            row[f'mean z2 {m}'] = (g.z ** 2).mean()
            row[f'rms rel err {m}'] = np.sqrt(np.average(g.rel_err ** 2, weights = g.exp_wins))
        rows.append(row)
    out += ['## Held-out calibration of each card\'s plain / upgraded copy, by training support\n', 'Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.\n', table(pd.DataFrame(rows))]

    # 3. Upgrade x combo
    cells = pd.read_csv(f'{d}/upgrade_cells.csv', keep_default_na = False, na_values = [''])
    ia, ib = f'model_interaction_{a}', f'model_interaction_{b}'
    rows = []
    for combo, g in cells.groupby('combo'):
        best, worst_ = g.loc[g.n_train.idxmax()], g.loc[g.n_train.idxmin()]
        rows.append({'combo': combo, 'kind': g.kind.iloc[0], 'n_best': best.n_train, 'n_rarest': worst_.n_train,
                     f'spread {a}': g[ia].max() - g[ia].min(), f'spread {b}': g[ib].max() - g[ib].min(),
                     f'best {a}': best[ia], f'rarest {a}': worst_[ia], f'best {b}': best[ib], f'rarest {b}': worst_[ib]})
    pc = pd.DataFrame(rows)
    sub = []
    for name, m in [('all pairs', pc.n_rarest >= 0), (f'rarest state combination <{RARE} rows', pc.n_rarest < RARE), ('rarest <1000 rows', pc.n_rarest < 1000)]:
        g = pc[m]
        # With a real interaction in the best supported states, how much of it shows up in the rarest ones?
        strong = g[g[f'best {a}'].abs() > STRONG]
        sub.append({'pair set': name, 'pairs': len(g), f'mean spread {a}': g[f'spread {a}'].mean(), f'mean spread {b}': g[f'spread {b}'].mean(),
                    'strong pairs': len(strong), f'rarest/best {a}': (strong[f'rarest {a}'] / strong[f'best {a}']).median() if len(strong) else np.nan,
                    f'rarest/best {b}': (strong[f'rarest {b}'] / strong[f'best {b}']).median() if len(strong) else np.nan})
    out += ['## Upgrade x combo: is a pair\'s interaction the same in every upgrade state?\n',
            'Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.\n',
            table(pd.DataFrame(sub))]
    top = pc.reindex(pc[f'spread {a}'].sub(pc[f'spread {b}']).abs().sort_values(ascending = False).index).head(8)
    out += ['Pairs whose spread changed most:\n', table(top[['combo', 'n_best', 'n_rarest', f'spread {a}', f'spread {b}', f'best {b}', f'rarest {b}']])]

    # 4. Combos, left over by the model
    c = pd.read_csv(f'{d}/combos.csv', keep_default_na = False, na_values = [''])
    both_up = set(cells.combo)
    c['has_upgrade_states'] = c.combo.isin(both_up)
    rows = []
    for name, g in [('listed combos', c[c.kind == 'synergy']), ('listed pairs with upgradable cards', c[(c.kind == 'synergy') & c.has_upgrade_states]),
                    ('listed, relics or triples', c[(c.kind == 'synergy') & ~c.has_upgrade_states]), ('null controls', c[c.kind == 'null'])]:
        row = {'set': name, 'n': len(g)}
        for m in (a, b):
            row[f'interaction {m}'] = g[f'model_interaction_{m}'].mean()
            row[f'resid train {m}'] = g[f'residual_train_{m}'].mean()
            row[f'resid test {m}'] = g[f'residual_test_{m}'].mean()
        rows.append(row)
    out += ['## Combos: model interaction and what it leaves unexplained (logit)\n', 'Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.\n', table(pd.DataFrame(rows))]
    key = c[c.combo.isin(['barricade + entrench', 'barricade + entrench + bodyslam', 'corruption + deadbranch', 'corruption + feelnopain', 'corruption + darkembrace'])]
    keep = ['combo', 'n_train', 'n_test'] + [f'{p}_{m}' for m in (a, b) for p in ('model_interaction', 'residual_train', 'residual_test')]
    out += ['Headline combos:\n', table(key[keep])]

    # 5. Cards and decisions
    for f, title in [('summary_combos', 'Combo summary'), ('summary_picks', 'Card picks')]:
        s = pd.read_csv(f'{d}/{f}.csv')
        s = s[s.model.isin([a, b])]
        out += [f'## {title}\n', table(s.T.reset_index().rename(columns = {'index': 'metric'}) if f == 'summary_combos' else s)]

    text = '\n'.join(out)
    with open(f'{d}/report_{b}_vs_{a}.md', 'w') as f:
        f.write(text)
    print(text)

if __name__ == '__main__':
    opts = dict(x.split('=', 1) for x in sys.argv[1:])
    main(opts['bench'], opts['control'], opts['treatment'])

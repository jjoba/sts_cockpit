'''
Do the five heads (act 1 boss, act 2 boss, act 3 first boss, act 3 second boss at A20, Heart kill) agree on which cards help? Run from the project root:

    python scripts/head_effects.py models=fnn_v24_ft_mt_hp_gold,fnn_v24_alpha_beta_mt_hp_gold [name=head_effects_v24] [data=training_data_v24]

For every card with 1,000+ picks (the card pick list of interaction_bench), a plain copy is added to held-out finetuning decks of the character that uses the card most
(colorless cards: all characters) that do not hold it, split by floor range (act 1 = floors 1-16, act 2 = 17-33, act 3 = 34+). The effect of a card on a head is the mean
change of that head's logit. Written to model_objects/bench/<name>/:
    head_effects_<model>.csv   one row per card x floor group: the effect on each head, plus the empirical effect (Mantel-Haenszel log odds ratio of the head's label for
                               rows holding the card versus not, stratified by floor, run on the held-out rows that have the label) and the rows behind it
    head_correlations.csv      correlation between heads (model effects), between model and empirical effects per head, by floor group, weighted by nothing
The point is to separate a real difference between heads from an artifact: the empirical effect is computed from the labels alone, no model.
'''

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interaction_bench as ib

HEADS = ['act1', 'act2', 'act3', 'act3b', 'heart']
GROUPS = {'act1': (1, 16), 'act2': (17, 33), 'act3': (34, 60)}
N_DECKS = 3000
MIN_PICKS = 1000
MIN_ROWS = 1500 # rows holding the card (with the head's label) needed for an empirical effect

@torch.no_grad()
def all_logits(net, X, batch = 100000):
    out = []
    if net.cols is None:
        net.use_layout()
    for i in range(0, len(X), batch):
        x = torch.as_tensor(np.asarray(X[i:i + batch]), dtype = torch.float32).to(ib.device).index_select(1, net.cols)
        if net.scale is not None:
            x = x * net.scale
        if net.base_idx is not None:
            x = x.index_add(1, net.base_idx, x.index_select(1, net.up_idx))
        for j, (W, b) in enumerate(net.layers):
            x = x @ W.T + b
            if j < len(net.layers) - 1:
                x = torch.relu(x)
        out.append(x.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)

def mh_log_or(has, label, stratum):
    # Mantel-Haenszel log odds ratio of label=1 for has=1 versus has=0 over strata (floors)
    k = int(stratum.max()) + 1
    a = np.bincount(stratum[has & (label == 1)], minlength = k).astype(float); b = np.bincount(stratum[has & (label == 0)], minlength = k).astype(float)
    c = np.bincount(stratum[~has & (label == 1)], minlength = k).astype(float); d = np.bincount(stratum[~has & (label == 0)], minlength = k).astype(float)
    n = np.maximum(a + b + c + d, 1)
    num, den = (a * d / n).sum(), (b * c / n).sum()
    return np.log(num / den) if num > 0 and den > 0 else np.nan

if __name__ == '__main__':
    opts = dict(a.split('=', 1) for a in sys.argv[1:])
    names = opts['models'].split(',')
    out_dir = f"{ib.ROOT}/model_objects/bench/{opts.get('name', 'head_effects')}"
    os.makedirs(out_dir, exist_ok = True)
    ib.configure(opts.get('data', 'training_data_v24'))
    nets = [ib.Net(n) for n in names]

    test = ib.open_split('test')
    rng = np.random.default_rng(0)
    # Held-out rows: features, extras, labels. The labels are the last 5 columns
    X = np.asarray(test[:, :ib.N_FEATURES + ib.N_EXTRA])
    Y = np.asarray(test[:, -ib.N_HEADS:])
    floor, char = X[:, ib.FLOOR].astype(int), X[:, 925:929].argmax(1)
    print(f'{len(X)} held-out rows')

    pick_cal = pd.read_csv(f'{ib.ROOT}/model_objects/bench/v24_picks/picks_calibration.csv')
    cards = sorted(pick_cal[pick_cal.picks >= MIN_PICKS].card.unique())
    print(f'{len(cards)} cards with {MIN_PICKS}+ picks')

    sub = np.sort(rng.choice(len(X), 400000, replace = False))
    rows = []
    for card in cards:
        plain, up = ib.item_cols(card)
        cols = [plain] + ([up] if up is not None else [])
        held = X[:, cols].sum(1) > 0
        share = np.array([held[char == c].mean() for c in range(4)])
        owner = share.argmax()
        chars = [owner] if share.max() > 3 * np.sort(share)[-2] or np.sort(share)[-2] == 0 else [0, 1, 2, 3]
        for g, (lo, hi) in GROUPS.items():
            in_g = (floor >= lo) & (floor <= hi) & np.isin(char, chars)
            cand = sub[in_g[sub] & ~held[sub]]
            if len(cand) < 200:
                continue
            idx = np.sort(rng.choice(cand, min(N_DECKS, len(cand)), replace = False))
            base = X[idx]
            added = base.copy()
            added[:, plain] += 1
            row = {'card': card, 'floor_group': g, 'owner': ib.CHARS[owner] if len(chars) == 1 else 'any', 'decks': len(idx)}
            for net in nets:
                d = all_logits(net, added) - all_logits(net, base)
                for h, hn in enumerate(HEADS):
                    row[f'{net.name}:{hn}'] = d[:, h].mean()
            # Empirical effect per head on the group's held-out rows of those characters
            m = in_g
            for h, hn in enumerate(HEADS):
                lab = Y[m, h]
                ok = lab >= 0
                has = held[m][ok]
                row[f'emp_rows_{hn}'] = int(has.sum())
                row[f'emp_{hn}'] = mh_log_or(has, lab[ok], floor[m][ok]) if has.sum() >= MIN_ROWS else np.nan
            rows.append(row)
        print(card, flush = True) if len(rows) % 40 == 0 else None
    df = pd.DataFrame(rows)

    corr_rows = []
    for net in nets:
        df.to_csv(f'{out_dir}/head_effects.csv', index = False)
        for g in list(GROUPS) + ['all']:
            d = df if g == 'all' else df[df.floor_group == g]
            for i, a in enumerate(HEADS):
                for b in HEADS[i + 1:]:
                    x, y = d[f'{net.name}:{a}'], d[f'{net.name}:{b}']
                    corr_rows.append({'model': net.name, 'floor_group': g, 'kind': 'model_vs_model', 'a': a, 'b': b, 'n': len(d), 'pearson': x.corr(y), 'spearman': x.corr(y, method = 'spearman')})
            for a in HEADS:
                dd = d.dropna(subset = [f'emp_{a}'])
                if len(dd) > 10:
                    corr_rows.append({'model': net.name, 'floor_group': g, 'kind': 'model_vs_empirical', 'a': a, 'b': 'empirical', 'n': len(dd),
                                      'pearson': dd[f'{net.name}:{a}'].corr(dd[f'emp_{a}']), 'spearman': dd[f'{net.name}:{a}'].corr(dd[f'emp_{a}'], method = 'spearman')})
    # Empirical effects of different heads against each other (no model)
    for g in list(GROUPS) + ['all']:
        d = df if g == 'all' else df[df.floor_group == g]
        for i, a in enumerate(HEADS):
            for b in HEADS[i + 1:]:
                dd = d.dropna(subset = [f'emp_{a}', f'emp_{b}'])
                if len(dd) > 10:
                    corr_rows.append({'model': 'none', 'floor_group': g, 'kind': 'empirical_vs_empirical', 'a': a, 'b': b, 'n': len(dd),
                                      'pearson': dd[f'emp_{a}'].corr(dd[f'emp_{b}']), 'spearman': dd[f'emp_{a}'].corr(dd[f'emp_{b}'], method = 'spearman')})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(f'{out_dir}/head_correlations.csv', index = False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print(corr.to_string(index = False))

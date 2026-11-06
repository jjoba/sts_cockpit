'''
Two checks on whether recommending by the Heart head hurts earlier survival, and whether the Heart head hides interactions the other heads see. Run from the project root:

    python scripts/head_tradeoff.py models=fnn_v24_alpha_beta_mt_hp_gold,fnn_v24_ft_mt_hp_gold [name=head_tradeoff_v24]

picks   Card reward decisions of A20 held-out runs (data/bench/decisions_v24_*.npy). For each decision the options are skip and each offered card. A recommender picks the
        option with the highest logit of one head (nearest boss of the act the reward is in, the A20 act 3 second boss = winning the run, or the Heart). The table gives, for
        what each recommender picks, the model's predicted nearest boss survival, run win and Heart probabilities, so the cost in earlier survival of following the Heart
        head is read off directly. Model based: it is only as good as the heads are. The player's actual pick is the reference row.
combos   Listed synergy combos (interaction_combos.py). Per head: the model interaction (counterfactual mixed difference on held-out decks of that character, floors 15-45,
        none of the cards) and the residual (logistic regression of that head's label on card presence with the head's own logit as offset, coefficient of all cards present,
        held-out rows at floor 20+ with a label for the head, run clustered). A residual above 0 = the head undervalues the combo.
'''

import os
import sys
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interaction_bench as ib
from head_effects import all_logits
from interaction_combos import COMBOS

HEADS = ['act1', 'act2', 'act3', 'act3b', 'heart']
N_PER_ACT = 60000
sigmoid = ib.sigmoid

def run_picks(nets, out_dir):
    X = np.load(f'{ib.ROOT}/data/bench/decisions_v24_X.npy', mmap_mode = 'r')
    M = np.load(f'{ib.ROOT}/data/bench/decisions_v24_M.npy')
    picked, offered, floor = M[:, 0].astype(int), M[:, 1:6].astype(int), M[:, 7].astype(int)
    rng = np.random.default_rng(0)
    acts = {'act1 (floors 1-16)': (1, 16, 0), 'act2 (17-33)': (17, 33, 1), 'act3 (34+)': (34, 60, 2)}
    rows = []
    for act, (lo, hi, boss) in acts.items():
        idx = np.flatnonzero((floor >= lo) & (floor <= hi))
        idx = np.sort(rng.choice(idx, min(N_PER_ACT, len(idx)), replace = False))
        base, opts, pk = np.asarray(X[idx]), offered[idx], picked[idx]
        n = len(idx)
        for net in nets:
            # Options: slot 0 = skip, slots 1-5 = offered cards. Logit of every head for each
            L = np.full((n, 6, 5), np.nan)
            L[:, 0] = all_logits(net, base)
            for k in range(5):
                v = opts[:, k]
                ok = np.flatnonzero(v >= 0)
                if len(ok) == 0:
                    continue
                x = base[ok].copy()
                x[np.arange(len(ok)), v[ok]] += 1
                L[ok, k + 1] = all_logits(net, x)
            # The player's actual option: skip = 0, else the slot holding the picked card
            actual = np.where(pk < 0, 0, 1 + (opts == pk[:, None]).argmax(1))
            P = sigmoid(L)
            r = np.arange(n)
            objectives = {'nearest boss': boss, 'run win (act 3 second boss)': 3, 'Heart kill': 4}
            choice = {name: np.nanargmax(np.where(np.isnan(L[:, :, h]), -1e9, L[:, :, h]), axis = 1) for name, h in objectives.items()}
            choice['player (actual)'] = actual
            choice['always skip'] = np.zeros(n, int)
            for name, c in choice.items():
                rows.append({'model': net.name, 'act': act, 'recommender': name, 'decisions': n, 'picks_a_card': float((c > 0).mean()),
                             'nearest_boss': P[r, c, boss].mean(), 'run_win': P[r, c, 3].mean(), 'heart': P[r, c, 4].mean(),
                             'same_as_heart_choice': float((c == choice['Heart kill']).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/picks_tradeoff.csv', index = False)
    pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(df.to_string(index = False))
    return df

def run_blend(nets, out_dir):
    # Decision rules that mix the nearest boss head and the Heart head: score = (1 - lam) * log P(nearest boss) + lam * log P(Heart) for each option.
    # lam is fixed (the frontier between the two), rises with the floor inside the act (floor weighted), or follows how likely the deck is to beat the boss (threshold)
    X = np.load(f'{ib.ROOT}/data/bench/decisions_v24_X.npy', mmap_mode = 'r')
    M = np.load(f'{ib.ROOT}/data/bench/decisions_v24_M.npy')
    picked, offered, floor = M[:, 0].astype(int), M[:, 1:6].astype(int), M[:, 7].astype(int)
    rng = np.random.default_rng(0)
    rows = []
    for act, (lo, hi, boss) in {'act1 (floors 1-16)': (1, 16, 0), 'act2 (17-33)': (17, 33, 1)}.items():
        idx = np.flatnonzero((floor >= lo) & (floor <= hi))
        idx = np.sort(rng.choice(idx, min(N_PER_ACT, len(idx)), replace = False))
        base, opts, pk, f = np.asarray(X[idx]), offered[idx], picked[idx], floor[idx]
        n = len(idx)
        for net in nets:
            L = np.full((n, 6, 5), np.nan)
            L[:, 0] = all_logits(net, base)
            for k in range(5):
                ok = np.flatnonzero(opts[:, k] >= 0)
                if len(ok) == 0:
                    continue
                x = base[ok].copy()
                x[np.arange(len(ok)), opts[ok, k]] += 1
                L[ok, k + 1] = all_logits(net, x)
            logp = -np.log1p(np.exp(-L)) # log sigmoid
            P = sigmoid(L)
            r = np.arange(n)
            progress = (f - lo) / (hi - lo)
            rules = {'nearest boss only (lam 0)': np.zeros(n), 'lam 0.25': np.full(n, 0.25), 'lam 0.5': np.full(n, 0.5), 'lam 0.75': np.full(n, 0.75), 'Heart only (lam 1)': np.ones(n),
                     'floor weighted: lam rises 0 to 1 across the act': progress, 'floor weighted, reversed: lam falls 1 to 0': 1 - progress,
                     'threshold: lam = min(1, P(boss now) / 0.5)': np.minimum(1, P[:, 0, boss] / 0.5), 'threshold: lam = min(1, P(boss now) / 0.35)': np.minimum(1, P[:, 0, boss] / 0.35)}
            for name, lam in rules.items():
                score = (1 - lam)[:, None] * logp[:, :, boss] + lam[:, None] * logp[:, :, 4]
                c = np.nanargmax(np.where(np.isnan(score), -1e9, score), axis = 1)
                rows.append({'model': net.name, 'act': act, 'rule': name, 'picks_a_card': float((c > 0).mean()), 'nearest_boss': P[r, c, boss].mean(), 'run_win': P[r, c, 3].mean(), 'heart': P[r, c, 4].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/blend_rules.csv', index = False)
    pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(df.to_string(index = False))
    return df

def run_combos(nets, out_dir):
    test = ib.open_split('test')
    Xall = np.asarray(test[:, :ib.N_FEATURES + ib.N_EXTRA])
    Y = np.asarray(test[:, -ib.N_HEADS:])
    floor, char = Xall[:, ib.FLOOR].astype(int), Xall[:, 925:929].argmax(1)
    drop = np.r_[False, floor[1:] < floor[:-1]]
    cid = np.cumsum(drop)
    keep = floor >= 20
    print(f'logits of {len(nets)} models on {len(Xall)} held-out rows', flush = True)
    Z = {net.name: all_logits(net, Xall) for net in nets}
    rng = np.random.default_rng(0)
    pool = np.flatnonzero((floor >= 15) & (floor <= 45))
    rows = []
    for c, lst in COMBOS.items():
        for items, why in lst:
            cols = [ib.item_cols(i) for i in items]
            held = np.stack([(Xall[:, p] > 0) | ((Xall[:, u] > 0) if u is not None else False) for p, u in cols], 1)
            in_char = (char == ib.CHARS.index(c)) & keep
            ctx_ok = pool[(char[pool] == ib.CHARS.index(c)) & ~held[pool].any(1)]
            ctx = Xall[rng.choice(ctx_ok, min(2000, len(ctx_ok)), replace = False)]
            for net in nets:
                plain_cols = [p for p, _ in cols]
                # mixed difference per head
                total = 0
                for s in range(len(plain_cols) + 1):
                    for sub in itertools.combinations(plain_cols, s):
                        x = ctx.copy()
                        if sub:
                            x[:, list(sub)] += 1
                        total = total + (-1) ** (len(plain_cols) - s) * all_logits(net, x)
                for h, hn in enumerate(HEADS):
                    ok = in_char & (Y[:, h] >= 0)
                    sel = np.flatnonzero(ok)
                    pres = held[sel]
                    subsets = [s for r_ in range(1, len(items) + 1) for s in itertools.combinations(range(len(items)), r_)]
                    D = np.stack([np.ones(len(sel))] + [pres[:, list(s)].prod(1) for s in subsets], 1).astype(float)
                    all_present = pres.all(1)
                    res = res_se = np.nan
                    if all_present.sum() >= 50:
                        b, se = ib.irls(Y[sel, h].astype(float), D, Z[net.name][sel, h], np.ones(len(sel)), cid[sel])
                        res, res_se = b[-1], se[-1]
                    rows.append({'model': net.name, 'char': c, 'combo': ' + '.join(items), 'size': len(items), 'head': hn, 'model_interaction': total[:, h].mean(),
                                 'residual': res, 'residual_se': res_se, 'rows_with_all': int(all_present.sum()), 'wins_with_all': int(Y[sel, h][all_present].sum())})
        print(c, 'done', flush = True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/combos_by_head.csv', index = False)
    s = df[df.rows_with_all >= 50].groupby(['model', 'size', 'head']).agg(combos = ('combo', 'count'), model_interaction = ('model_interaction', 'mean'),
                                                                           residual = ('residual', 'mean')).reset_index()
    s.to_csv(f'{out_dir}/summary_combos_by_head.csv', index = False)
    print(s.to_string(index = False))
    return df

if __name__ == '__main__':
    opts = dict(a.split('=', 1) for a in sys.argv[1:])
    names = opts['models'].split(',')
    out_dir = f"{ib.ROOT}/model_objects/bench/{opts.get('name', 'head_tradeoff')}"
    os.makedirs(out_dir, exist_ok = True)
    ib.configure(opts.get('data', 'training_data_v24'))
    nets = [ib.Net(n) for n in names]
    for sec in opts.get('sections', 'picks,combos').split(','):
        print('\n=====', sec)
        {'picks': run_picks, 'combos': run_combos, 'blend': run_blend}[sec](nets, out_dir)
